import os
import time
import pandas as pd
import numpy as np
import tempfile
import zipfile
import tarfile
from threading import Thread
from queue import Queue
from autogluon.tabular import TabularPredictor
import streamlit as st
from autogluon.core.models import AbstractModel
from autogluon.features.generators import LabelEncoderFeatureGenerator
from autogluon.core.space import Real, Int, Categorical, Space
from sklearn.svm import SVR
from concurrent.futures import ThreadPoolExecutor

class CustomSVRModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._feature_generator = None

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        return X.fillna(0).to_numpy(dtype=np.float32)

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        from sklearn.svm import SVR
        model_cls = SVR
        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        self.model = model_cls(**params)
        self.model.fit(X, y)

    def _set_default_params(self):
        default_params = {
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'C': 1.0,
            'epsilon': 0.1,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            valid_raw_types=['int', 'float', 'category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params
    

    def _get_default_searchspace(self) -> Space:
        return Space(
            kernel=Categorical('linear', 'poly', 'rbf', 'sigmoid'),
            degree=Int(lower=1, upper=5),
            gamma=Categorical('scale', 'auto'),
            C=Real(lower=0.1, upper=2.0, default=1.0, log=True),
            epsilon=Real(lower=0.01, upper=0.2, default=0.1, log=True),
        )
    
# サイドバーにアップロード用のUIを作成し、アップロードされたファイルを返す関数
def upload_file(header, types):
    st.sidebar.header(header)
    return st.sidebar.file_uploader(f"Choose a {header}", type=types)

# データフレームを表示し、ダウンロード可能な形で提供する関数
def show_dataframe(df, header, download_name):
    st.header(header)
    st.dataframe(df)
    st.download_button(f"Download {header}", df.to_csv(index=False), file_name=download_name)

# 入力データが正しいか検証する関数
def validate_input(df, target_variable):
    if target_variable not in df.columns:
        raise ValueError(f"The target variable '{target_variable}' is not found in the dataset.")
    
# リザルトデータを表示し、ダウンロード可能な形で提供する関数
def display_results(header, data, file_name):
    if st.button(f'Show {header}'):
        st.header(header)
        st.write(data)
        if isinstance(data, dict):
            data = pd.DataFrame.from_dict(data, orient='index')
            st.download_button(f"Download {header}", data.to_csv(index=True), file_name=file_name)
        else:
            st.download_button(f"Download {header}", data.to_csv(index=True), file_name=file_name)

# モデルをトレーニングする関数
def train_model(data, target_variable, time_limit, groups, presets, models):
    groups = None if groups == "" else groups
    hyperparameters = {model: {} for model in models}  # 選択されたモデルだけを指定
    return TabularPredictor(label=target_variable, groups=groups).fit(train_data=data, time_limit=time_limit, presets=presets, hyperparameters=hyperparameters)

def train_model_thread(data, target_variable, time_limit, groups, presets, models, queue):
    result = {'status': 'success'}
    with ThreadPoolExecutor(max_workers=2) as executor:  # スレッドの最大数を2と設定
        future = executor.submit(train_model, data, target_variable, time_limit, groups, presets, models)
        try:
            predictor = future.result()
            result['path'] = save_model(predictor)
        except Exception as e:
            result['status'] = 'failure'
            result['error'] = str(e)
    queue.put(result)

# トレーニングしたモデルを保存する関数
def save_model(predictor):
    model_dir = predictor.path
    zip_path = f'{model_dir}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), os.path.join(model_dir, '..')))
    return zip_path

def train_model_and_save_result(data, target_variable, time_limit, groups, presets, models):
    with st.spinner('Training...'):
        queue = Queue()
        thread = Thread(target=train_model_thread, args=(data, target_variable, time_limit, groups, presets, models, queue))
        thread.start()
    while thread.is_alive():
        st.write('Training in progress...')
        st.progress(50)
        time.sleep(1)
    result = queue.get()  # queueから結果を取得
    if result['status'] == 'success':
        zip_path = result['path']
        st.success(f'Training completed! Model saved at {zip_path}')
        with open(zip_path, 'rb') as f:
            bytes = f.read()
            st.header('Model Download')
            st.download_button(label="Download model", data=bytes, file_name='model.zip')
    else:
        st.error(f'Training failed due to error: {result["error"]}')


def load_model_and_make_prediction(model_file, test_file):
    if test_file is not None:
        test_data = pd.read_csv(test_file)

        if model_file is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    if model_file.name.endswith('.zip'):
                        with zipfile.ZipFile(model_file, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                    else:
                        with tarfile.open(fileobj=model_file) as tar:
                            tar.extractall(path=temp_dir)
                    model_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])  # Extract the first directory of the extracted files
                    predictor = TabularPredictor.load(model_dir, verbosity=2)
                    performance = predictor.evaluate(test_data)
                    performance_df = pd.DataFrame.from_dict(performance, orient='index', columns=['Performance'])
                    predictions = predictor.predict(test_data)
                    feature_importance = predictor.feature_importance(test_data)
                    leaderboard = predictor.leaderboard(test_data)

                    # Display and make downloadable various results
                    display_results('Model Performance', performance_df, 'model_performance.csv')
                    display_results('Feature Importance', feature_importance, 'feature_importance.csv')
                    display_results('Model Predictions', predictions, 'model_predictions.csv')

                    # Add prediction results to test data and display it
                    if st.button('Show Test Data with Predictions'):
                        st.header('Test Data with Predictions')
                        test_data['Predictions'] = predictions
                        st.dataframe(test_data)
                        st.download_button("Download Test Data with Predictions", test_data.to_csv(index=True), file_name='test_data_with_predictions.csv')

                    display_results('Leaderboard', leaderboard, 'leaderboard.csv')

                # Error handling in case of problems with model loading or prediction
                except Exception as e:
                    st.error(f'Error during loading model or making predictions: {e}')

# モデルをトレーニングする関数
def train_model(data, target_variable, time_limit, groups, presets, models):
    groups = None if groups == "" else groups
    hyperparameters = {model: {} for model in models}  # 選択されたモデルだけを指定
    return TabularPredictor(label=target_variable, groups=groups).fit(train_data=data, time_limit=time_limit, presets=presets, hyperparameters=hyperparameters)

# アプリケーションのエントリーポイント
def app():
    st.title('AutoML App')

    # データファイルのアップロード
    data_file = upload_file('Upload CSV', 'csv')
    if data_file is not None:
        data = pd.read_csv(data_file)
        show_dataframe(data, 'Training Data Preview', 'training_data.csv')

        target_variable = st.sidebar.text_input('Target Variable')
        try:
            validate_input(data, target_variable)
        except ValueError as e:
            st.error(f'Invalid input: {e}')
            return

        time_limit = st.sidebar.number_input('Time Limit', value=3, step=1)
        groups = st.sidebar.text_input('Groups', help='This field is optional. Leave it blank if you do not want to specify any groups.')
        presets = st.sidebar.selectbox(
            'Presets',
            ('best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'interpretable'),
            help='Select the presets for the model training.'
        )

        # モデル選択UIの追加
        model_selection = {
            'GBM': st.sidebar.checkbox('LightGBM'),
            'CAT': st.sidebar.checkbox('CatBoost'),
            'XGB': st.sidebar.checkbox('XGBoost'),
            'NN_TORCH': st.sidebar.checkbox('Neural network'),
            'RF': st.sidebar.checkbox('Random Forest'),
            'XT': st.sidebar.checkbox('Extra Trees'),
            'KNN': st.sidebar.checkbox('k-Nearest Neighbors'),
            CustomSVRModel: st.sidebar.checkbox('CustomSVRModel'),
        }
        selected_models = [model for model, selected in model_selection.items() if selected]

        if st.sidebar.button('Train Model'):
            train_model_and_save_result(data, target_variable, time_limit, groups, presets, selected_models)

    # モデルファイルとテストデータのアップロード
    model_file = upload_file('Upload Model', ["zip"])
    test_file = upload_file('Upload Test CSV', 'csv')

    load_model_and_make_prediction(model_file, test_file)

# アプリケーションのエントリーポイント
if __name__ == '__main__':
    app()