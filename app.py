import re,jwt,os,logging,traceback,joblib,numpy as np,pandas as pd
from flask import Flask,jsonify, render_template, request,redirect, url_for
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import plotly.graph_objects as go
from collections import defaultdict
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "logistic_split_70.pkl")

model = joblib.load(MODEL_PATH)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    user_data = users_collection.find_one({'_id': ObjectId(user_id)})
    if user_data:
        return User(user_data)
    return None

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
def serialize_user_data(user_data):
    """Convert non-serializable fields in user_data to serializable types."""
    if user_data:
        user_data['_id'] = str(user_data['_id'])  
        if 'created_at' in user_data and isinstance(user_data['created_at'], datetime):
            user_data['created_at'] = user_data['created_at'].isoformat()  #
    return user_data

# MongoDB setup
client = MongoClient(os.environ.get('MONGODB_URI'))
db = client.mindscope_db
users_collection = db.users
password_reset_collection = db.password_resets
# Dataset path
DATASET_PATH = "data/Depression Student Dataset.csv"
# User class
class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.password_hash = user_data['password_hash'] 
    def get_id(self):
        return self.id   
    @property
    def is_active(self):
        return self.user_data.get('is_active', True)
    @property
    def is_authenticated(self):
        return True
    @property
    def is_anonymous(self):
        return False
# ML Models dictionary
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(probability=True),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}
class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = None
        self.numerical_columns = None
        self.target_encoder = LabelEncoder()
    def fit(self, df, y=None):
        self.categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        self.numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for column in self.categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            self.label_encoders[column].fit(df[column].astype(str))
        if len(self.numerical_columns) > 0:
            self.scaler.fit(df[self.numerical_columns])
        if y is not None:
            self.target_encoder.fit(y) 
        return self
    def transform(self, df):
        df_transformed = df.copy()
        for column in self.categorical_columns:
            df_transformed[column] = self.label_encoders[column].transform(df_transformed[column].astype(str))
        
        if len(self.numerical_columns) > 0:
            df_transformed[self.numerical_columns] = self.scaler.transform(df_transformed[self.numerical_columns])
        
        return df_transformed
    
    def transform_target(self, y):
        return self.target_encoder.transform(y)
    
    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)
    
    def get_feature_names(self):
        return list(self.categorical_columns) + list(self.numerical_columns)
    
    def get_categorical_values(self, column):
        if column in self.categorical_columns:
            return self.label_encoders[column].classes_.tolist()
        return None

preprocessor = DataPreprocessor()

# Utility functions
def validate_password(password):
    """Validate password strength and return detailed feedback."""
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search("[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search("[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search("[0-9]", password):
        errors.append("Password must contain at least one number")
    if not re.search("[^a-zA-Z0-9]", password):
        errors.append("Password must contain at least one special character")
    
    if errors:
        return False, errors
    return True, "Password is valid"
# ML Helper functions
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model and return performance metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics

def get_best_split(model_name, X, y):
    split_ratios = [0.1, 0.2, 0.3, 0.4]
    results = []
    X_processed = preprocessor.fit_transform(X)
    y_encoded = preprocessor.transform_target(y)
    for split in split_ratios:
        scores = []
        for _ in range(5):
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=split, random_state=None)
            metrics = evaluate_model(models[model_name], X_train, X_test, y_train, y_test)
            scores.append(metrics)
        avg_metrics = {
            key: np.mean([score[key] for score in scores])
            for key in scores[0].keys()
        }
        results.append({
            'split': split,
            **avg_metrics
        })
    best_split = max(results, key=lambda x: x['f1'])
    return results, best_split

def compare_all_models(X, y):
    """Compare all models and return their performance metrics."""
    split_ratios = [0.1, 0.2, 0.3, 0.4]
    all_results = defaultdict(list)
    best_model_info = {
        'model': None,
        'split': None,
        'metrics': None,
        'avg_f1': -1
    }
    
    X_processed = preprocessor.fit_transform(X)
    y_encoded = preprocessor.transform_target(y)
    
    for model_name, model in models.items():
        model_f1_scores = []
        for split in split_ratios:
            scores = []
            for _ in range(5):  # Run 5 iterations for each split ratio
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=split, random_state=None
                )
                metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
                scores.append(metrics)
            
            avg_metrics = {
                key: np.mean([score[key] for score in scores])
                for key in scores[0].keys()
            }
            all_results[model_name].append({
                'split': split,
                **avg_metrics
            })
            model_f1_scores.append(avg_metrics['f1'])
        
        avg_f1 = np.mean(model_f1_scores)
        if avg_f1 > best_model_info['avg_f1']:
            best_split_idx = np.argmax(model_f1_scores)
            best_model_info = {
                'model': model_name,
                'split': split_ratios[best_split_idx],
                'metrics': all_results[model_name][best_split_idx],
                'avg_f1': avg_f1
            }
    
    return all_results, best_model_info

def generate_model_explanation(best_model_info):
    model_name = best_model_info['model']
    metrics = best_model_info['metrics']
    
    strengths = {
        'Logistic Regression': 'simple, interpretable, and efficient for linearly separable data',
        'Decision Tree': 'handles non-linear relationships and is easy to interpret',
        'Random Forest': 'reduces overfitting and handles complex relationships',
        'Gradient Boosting': 'typically provides high accuracy and handles imbalanced data well',
        'SVM': 'effective in high-dimensional spaces and handles non-linear relationships',
        'KNN': 'simple and effective for pattern recognition',
        'Naive Bayes': 'works well with high-dimensional data and is computationally efficient',
        'XGBoost': 'optimized implementation of gradient boosting with high performance'
    }
    explanation = f"""
    Based on comprehensive testing across multiple split ratios, the {model_name} emerged as the best model for this depression prediction task.
    Key Performance Metrics:
    - F1 Score: {metrics['f1']:.3f}
    - Accuracy: {metrics['accuracy']:.3f}
    - Precision: {metrics['precision']:.3f}
    - Recall: {metrics['recall']:.3f}
    Optimal Split Ratio: {best_model_info['split']}
    Why this model performs best:
    1. {strengths[model_name]}
    2. It achieves the best balance between precision and recall (F1 score)
    3. The model shows consistent performance across different train-test splits
    """
    return explanation

# Routes
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

@app.route('/signup', methods=['GET'])
def signup_page():
    return render_template('signup.html')

@app.route('/auth/logout', methods=['POST'])
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/auth/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'password', 'dateOfBirth', 'gender']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # Check for existing user
        existing_user = users_collection.find_one({'email': data['email'].lower()})
        if existing_user:
            return jsonify({'error': 'Email already registered'}), 400

        # Validate email format
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not email_pattern.match(data['email']):
            return jsonify({'error': 'Invalid email format'}), 400

        # Validate password strength
        password_validation, message = validate_password(data['password'])
        if not password_validation:
            return jsonify({'error': message}), 400

        # Validate date format and range
        try:
            dob = datetime.strptime(data['dateOfBirth'], '%Y-%m-%d')
            if dob > datetime.now():
                return jsonify({'error': 'Date of birth cannot be in the future'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        # Validate gender
        allowed_genders = ['male', 'female', 'non-binary', 'other', 'prefer-not']
        if data['gender'].lower() not in allowed_genders:
            return jsonify({'error': 'Invalid gender value'}), 400

        # Add security questions and answers
        if not data.get('security_questions') or len(data['security_questions']) != 2:
            return jsonify({'error': 'You must provide exactly 2 security questions and answers'}), 400

        # Sanitize inputs
        user_data = {
            'firstName': str(data['firstName']).strip()[:50],
            'lastName': str(data['lastName']).strip()[:50],
            'email': data['email'].lower().strip(),
            'password_hash': generate_password_hash(data['password'], method='pbkdf2:sha256:260000'),
            'dateOfBirth': dob,
            'gender': data['gender'].lower(),
            'phone': data.get('phone'),
            'security_questions': data['security_questions'],  # Store security questions and answers
            'created_at': datetime.utcnow(),
            'profile_completed': True
        }

        # Insert user
        result = users_collection.insert_one(user_data)

        return jsonify({
            'message': 'Account created successfully',
            'user_id': str(result.inserted_id)
        }), 201

    except Exception as e:
        logger.error(f"Unexpected error during signup: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500   

@app.route('/auth/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            print("Received data:", data)  # Log the received data
            if not data or not data.get('email') or not data.get('password'):
                return jsonify({'error': 'Missing email or password'}), 400
            
            user_data = users_collection.find_one({'email': data['email'].lower()})
            print("User data from DB:", user_data)  # Log the user data retrieved from the database
            
            if not user_data:
                return jsonify({'error': 'Invalid email or password'}), 401
            
            if check_password_hash(user_data['password_hash'], data['password']):
                user = User(user_data)
                login_user(user)
                next_page = request.args.get('next')
                print("Login successful, redirecting to:", next_page or url_for('explore'))  # Log the redirect URL

                # Serialize user_data to make it JSON serializable
                serialized_user_data = serialize_user_data(user_data)

                return jsonify({
                    'message': 'Logged in successfully',
                    'user': serialized_user_data,
                    'redirect': next_page or url_for('explore')
                })
            
            return jsonify({'error': 'Invalid email or password'}), 401
        
        except Exception as e:
            print(f"Server Error: {str(e)}")  # Check your server logs for this message
            return jsonify({'error': 'Internal server error. Please try again.'}), 500
        
@app.route('/forgot-password', methods=['GET'])
def forgot_password_page():
    return render_template('forgot-password.html')

# Forgot Password - Retrieve Security Questions
@app.route('/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')

    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Check if the user exists
    user = users_collection.find_one({'email': email.lower()})
    if not user:
        return jsonify({'error': 'No account found with this email'}), 404

    # Return the user's security questions
    security_questions = user.get('security_questions', [])
    if not security_questions:
        return jsonify({'error': 'No security questions found for this user'}), 400

    return jsonify({
        'message': 'Security questions retrieved',
        'security_questions': [q['question'] for q in security_questions]  # Return only questions
    }), 200
# Verify Security Answers
@app.route('/verify-security-answers', methods=['POST'])
def verify_security_answers():
    data = request.get_json()
    email = data.get('email')
    submitted_answers = data.get('answers', [])

    if not email or not submitted_answers:
        return jsonify({'error': 'Email and answers are required'}), 400

    # Check if the user exists
    user = users_collection.find_one({'email': email.lower()})
    if not user:
        return jsonify({'error': 'No account found with this email'}), 404

    # Verify the answers
    security_questions = user.get('security_questions', [])
    if not security_questions:
        return jsonify({'error': 'No security questions found for this user'}), 400

    # Add logging to help debug
    print("Submitted answers:", submitted_answers)
    print("Stored questions:", security_questions)

    # Compare answers (case-insensitive)
    try:
        matches = all(
            submitted.get('answer', '').strip().lower() == stored['answer'].strip().lower()
            for submitted, stored in zip(submitted_answers, security_questions)
        )
        
        if matches:
            # Generate a temporary token for password reset using PyJWT
            reset_token = jwt.encode(
                payload={
                    'email': email, 
                    'exp': datetime.utcnow() + timedelta(minutes=10)
                },
                key=app.config['SECRET_KEY'],
                algorithm='HS256'
            )
            
            # If using PyJWT >= 2.0.0, encode() returns a string
            # If using an older version, you might need to decode bytes to string
            if isinstance(reset_token, bytes):
                reset_token = reset_token.decode('utf-8')
                
            return jsonify({
                'message': 'Security answers verified',
                'reset_token': reset_token
            }), 200
        else:
            return jsonify({'error': 'Incorrect answers to security questions'}), 401
            
    except Exception as e:
        print("Verification error:", str(e))
        return jsonify({'error': 'Error verifying answers: ' + str(e)}), 500
    
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    reset_token = data.get('reset_token')
    new_password = data.get('new_password')

    if not reset_token or not new_password:
        return jsonify({'error': 'Reset token and new password are required'}), 400

    try:
        # Verify the reset token
        payload = jwt.decode(reset_token, app.config['SECRET_KEY'], algorithms=['HS256'])
        email = payload.get('email')

        # Check if the user exists
        user = users_collection.find_one({'email': email.lower()})
        if not user:
            return jsonify({'error': 'No account found with this email'}), 404

        # Validate the new password
        password_validation, message = validate_password(new_password)
        if not password_validation:
            return jsonify({'error': message}), 400

        # Update the password
        result = users_collection.update_one(
            {'email': email.lower()},
            {'$set': {'password_hash': generate_password_hash(new_password)}}
        )

        # Check if the update was successful
        if result.modified_count == 0:
            return jsonify({'error': 'Failed to update password. Please try again later.'}), 500

        return jsonify({'message': 'Password reset successfully'}), 200

    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Reset token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid reset token'}), 401
    except Exception as e:
        print(f"Error resetting password: {str(e)}")
        return jsonify({'error': 'Failed to reset password. Please try again later.'}), 500
    
@app.route('/selection')
def model_selection():
    return render_template('model_selection.html', models=list(models.keys()))

@app.route('/performance/<model_name>')
def model_performance(model_name):
    df = pd.read_csv(DATASET_PATH)
    X = df.drop('Depression', axis=1)
    y = df['Depression']
    
    preprocessor.fit(X, y)
    feature_names = preprocessor.get_feature_names()
    categorical_values = {
        column: preprocessor.get_categorical_values(column)
        for column in preprocessor.categorical_columns
    }
    
    results, best_split = get_best_split(model_name, X, y)
    
    fig = go.Figure()
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=[r['split'] for r in results],
            y=[r[metric] for r in results],
            name=metric.capitalize(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title=f'{model_name} Performance Across Different Split Ratios',
        xaxis_title='Test Split Ratio',
        yaxis_title='Score',
        yaxis_range=[0, 1]
    )
    
    explanation = f"""
    The best split ratio for {model_name} is {best_split['split']} with:
    - F1 Score: {best_split['f1']:.3f}
    - Accuracy: {best_split['accuracy']:.3f}
    - Precision: {best_split['precision']:.3f}
    - Recall: {best_split['recall']:.3f}
    """
    
    return render_template(
        'model_performance.html',
        model_name=model_name,
        plot=fig.to_html(full_html=False),
        best_split=best_split,
        explanation=explanation,
        feature_names=feature_names,
        categorical_values=categorical_values
    )

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        df = pd.read_csv(DATASET_PATH)
        X = df.drop('Depression', axis=1)
        y = df['Depression']
        
        form_data = request.form.to_dict()
 
        if not form_data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        if not hasattr(preprocessor, 'label_encoders') or not preprocessor.label_encoders:
            preprocessor.fit(X, y)
        
        ordered_features = [
            'Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 
            'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Study Hours', 'Financial Stress', 'Family History of Mental Illness'
        ]
        
        ordered_data = {feature: form_data[feature] for feature in ordered_features}
        input_df = pd.DataFrame([ordered_data])
        
        try:
            input_processed = preprocessor.transform(input_df)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return jsonify({'success': False, 'error': 'Invalid input data format'}), 400

        X_processed = preprocessor.transform(X)
        y_encoded = preprocessor.transform_target(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, 
            test_size=0.2,
            random_state=42
        )

        model = models[model_name]
        model.fit(X_train, y_train)
        prediction_prob = model.predict_proba(input_processed)[0][1]
        
        return jsonify({
            'success': True,
            'depression_percentage': float(prediction_prob * 100),
            'prediction': "Yes" if prediction_prob >= 0.5 else "No"
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/justification')
def model_justification():
    df = pd.read_csv(DATASET_PATH)
    X = df.drop('Depression', axis=1)
    y = df['Depression']
    
    all_results, best_model_info = compare_all_models(X, y)
    
    fig = go.Figure()
    for model_name, results in all_results.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=[f"Split {r['split']}" for r in results],
            y=[r['f1'] for r in results],
            text=[f"{r['f1']:.3f}" for r in results],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison Across Different Splits',
        xaxis_title='Test Split Ratio',
        yaxis_title='F1 Score',
        yaxis_range=[0, 1],
        barmode='group',
        height=600
    )
    
    explanation = generate_model_explanation(best_model_info)
    
    return render_template(
        'model_justification.html',
        plot=fig.to_html(full_html=False),
        best_model=best_model_info,
        explanation=explanation
    )

@app.route('/dataset')
def dataset_visualization():
    df = pd.read_csv(DATASET_PATH)
    
    dataset_info = {
        'num_rows': len(df),
        'num_features': len(df.columns),
        'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    feature_plots = {}
    for feature in df.columns:
        fig = go.Figure()
        if df[feature].dtype in ['int64', 'float64']:
            fig.add_trace(go.Histogram(x=df[feature], name="Distribution"))
            fig.add_trace(go.Box(x=df[feature], name="Box Plot"))
            fig.update_layout(title=f'Distribution and Box Plot of {feature}')
        else:
            value_counts = df[feature].value_counts()
            fig.add_trace(go.Bar(x=value_counts.index, y=value_counts.values))
            fig.update_layout(title=f'Distribution of {feature}')
        
        feature_plots[feature] = fig.to_html(full_html=False)
    
    correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()
    correlation_plot = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    correlation_plot.update_layout(title='Feature Correlation Heatmap')
    
    return render_template(
        'dataset_visualization.html',
        dataset_info=dataset_info,
        feature_plots=feature_plots,
        correlation_plot=correlation_plot.to_html(full_html=False)
    )

@app.route('/assessment', methods=['GET'])
@login_required
def assessment_page():
    print(f"Current User: {current_user.id}")  # Debugging: Print the current user's ID
    return render_template('assessment.html')

@app.route('/assessment-prediction', methods=['POST'])
@login_required
def assessment_prediction():
    try:
        # Get form data from the frontend
        form_data = request.get_json()
        
        # Check if form data is empty
        if not form_data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        # Define the expected features in the correct order
        ordered_features = [
            'Gender', 'Age', 'Academic Pressure', 'Study Satisfaction', 
            'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
            'Study Hours', 'Financial Stress', 'Family History of Mental Illness'
        ]
        
        # Ensure the form data matches the expected features
        ordered_data = {feature: form_data.get(feature) for feature in ordered_features}
        
        # Convert the ordered data into a DataFrame
        input_df = pd.DataFrame([ordered_data])
        
        # Preprocess the input data using the preprocessor
        try:
            input_processed = preprocessor.transform(input_df)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error in preprocessing: {str(e)}'}), 400
        
        # Make prediction using the trained model
        prediction_prob = model.predict_proba(input_processed)[0][1]  # Assuming binary classification
        
        # Return the prediction result
        return jsonify({
            'success': True,
            'depression_percentage': float(prediction_prob * 100),
            'prediction': "Yes" if prediction_prob >= 0.5 else "No"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/explore')
@login_required
def explore():
    return render_template('explore.html')
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

def create_indexes():
    users_collection.create_index('email', unique=True)
    users_collection.create_index('phone')
    password_reset_collection.create_index('created_at', expireAfterSeconds=3600)
    db.activity_logs.create_index('timestamp')
    db.activity_logs.create_index('user_id')

if __name__ == '__main__':
    try:
        df = pd.read_csv(DATASET_PATH)
        logger.info(f"Successfully loaded dataset with {len(df)} samples and {len(df.columns)} features")   
        X = df.drop('Depression', axis=1)
        y = df['Depression']
        preprocessor.fit(X, y)
        logger.info("Successfully initialized preprocessor")
        create_indexes()
        app.run(debug=True)
    except Exception as e:
        logger.error(f"error")
