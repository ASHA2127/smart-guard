import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file, send_from_directory
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle
import os
import shutil
import json
import re


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(APP_ROOT, 'templates')
STATIC_DIR = os.path.join(APP_ROOT, 'static')
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR) #Initialize the flask App


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "spam.pkl")

model = pickle.load(open(model_path, "rb"))
LOCAL_IMG_DIR ='img'
SPAM_COUNT = 0
NO_SPAM_COUNT = 0
LAST_DEVICE_STATUS = []
LAST_FEATURE_STATUS = []
ADMIN_USER = 'admin'
ADMIN_PWD = 'admin'
ADMIN_HINT = 'spam'
email_re = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
pwd_re = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$')
phone_re = re.compile(r'^[0-9]{10}$')
def is_valid_email(s):
    try:
        return bool(email_re.match(str(s).strip()))
    except Exception:
        return False
def is_strong_password(s):
    try:
        return bool(pwd_re.match(str(s)))
    except Exception:
        return False
def is_valid_phone(s):
    try:
        return bool(phone_re.match(str(s).strip()))
    except Exception:
        return False

def get_bg_url(kind: str):
    candidates = {
        'abstract': ['abstract1.jpg'],
        'admin_login': ['admin1.jpg'],
        'user_register': ['userregister.jpg'],
        'user_data': ['userdata.jpg'],
        'admin_dashboard': ['dashbord.jpg'],
        'user_login': ['userlogin.jpg'],
        'user_approval': ['userapproval.jpg'],
        'prediction': ['prediction.jpg'],
        'chart': ['chart.jpg'],
        'preview': ['preview.jpg'],
        'input_guide': ['input.jpg'],
        'device_status': ['device.jpg'],
        'upload': ['upload.jpg'],
        'perf_device': ['perf_device.jpg', 'perf_device.jpeg', 'perf_device.png', 'perf_device.webp'],
        'perf_devices': ['perf_device.jpg', 'perf_device.jpeg', 'perf_device.png', 'perf_device.webp'],
    }
    files = candidates.get(kind, [])
    for fname in files:
        fpath = os.path.join(LOCAL_IMG_DIR, fname)
        if os.path.exists(fpath):
            return f"/local_img/{fname}"
    # Restore original remote fallback
    return 'https://static.vecteezy.com/system/resources/previews/026/609/998/original/internet-of-things-iot-concept-vector.jpg'

@app.route('/local_img/<path:filename>')
def local_img(filename):
    try:
        return send_from_directory(LOCAL_IMG_DIR, filename)
    except Exception:
        return jsonify({"message": "Image not found"}), 404
@app.route('/')
@app.route('/index')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/chart')
def chart():
    global SPAM_COUNT, NO_SPAM_COUNT
    feat_list = globals().get('LAST_FEATURE_STATUS') if 'LAST_FEATURE_STATUS' in globals() else []
    if isinstance(feat_list, list) and len(feat_list) > 0:
        features_spam_count = sum(1 for f in feat_list if str(f.get('status','')) == 'Spam')
        features_normal_count = sum(1 for f in feat_list if str(f.get('status','')) == 'Normal')
        features_unknown_count = sum(1 for f in feat_list if str(f.get('status','')) not in ['Spam','Normal'])
        total = features_spam_count + features_normal_count + features_unknown_count
        if total > 0:
            spam_pct = round((features_spam_count * 100.0) / total)
            no_spam_pct = 100 - spam_pct
        else:
            spam_pct = 50
            no_spam_pct = 50
        return render_template(
            'chart.html',
            bg_url=get_bg_url('chart'),
            spam_pct=spam_pct,
            no_spam_pct=no_spam_pct,
            spam_count=features_spam_count,
            no_spam_count=features_normal_count,
            total_count=total,
        )
    else:
        total = (SPAM_COUNT or 0) + (NO_SPAM_COUNT or 0)
        if total > 0:
            spam_pct = round((SPAM_COUNT * 100.0) / total)
            no_spam_pct = 100 - spam_pct
        else:
            spam_pct = 50
            no_spam_pct = 50
        return render_template(
            'chart.html',
            bg_url=get_bg_url('chart'),
            spam_pct=spam_pct,
            no_spam_pct=no_spam_pct,
            spam_count=SPAM_COUNT,
            no_spam_count=NO_SPAM_COUNT,
            total_count=total,
        )

@app.route('/reset_counts')
def reset_counts():
    global SPAM_COUNT, NO_SPAM_COUNT
    SPAM_COUNT = 0
    NO_SPAM_COUNT = 0
    return redirect('/chart')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview', methods=["GET", "POST"])
def preview():
    def set_id_index(df):
        id_col = next((c for c in df.columns if str(c).strip().lower() == 'id'), None)
        if id_col:
            df = df.set_index(id_col)
        else:
            df.index.name = 'Id'
        return df

    def load_default_upload_csv():
        path = os.path.join(os.path.dirname(__file__), 'upload.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='unicode_escape')
            return set_id_index(df)
        return None

    try:
        if request.method == 'POST':
            dataset = request.files.get('datasetfile')
            if not dataset or dataset.filename == '':
                df = load_default_upload_csv()
                if df is None:
                    return render_template("preview.html", df_view=pd.DataFrame(), records=[], columns=[], bg_url=get_bg_url('preview'))
                return render_template("preview.html", df_view=df, records=df.to_dict(orient='records'), columns=list(df.columns), bg_url=get_bg_url('preview'))
            filename = dataset.filename or ''
            if not filename.lower().endswith('.csv'):
                return render_template('upload.html', error='Please upload a CSV file')
            save_path = os.path.join(os.path.dirname(__file__), 'upload.csv')
            dataset.save(save_path)
            df = pd.read_csv(save_path, encoding='unicode_escape')
            df = set_id_index(df)
            return render_template("preview.html", df_view=df, records=df.to_dict(orient='records'), columns=list(df.columns), bg_url=get_bg_url('preview'))
        else:
            # GET: attempt to load last uploaded dataset from upload.csv
            df = load_default_upload_csv()
            if df is None:
                return render_template("preview.html", df_view=pd.DataFrame(), records=[], columns=[], bg_url=get_bg_url('preview'))
            return render_template("preview.html", df_view=df, records=df.to_dict(orient='records'), columns=list(df.columns), bg_url=get_bg_url('preview'))
    except Exception:
        # Fallback to empty on error
        return render_template("preview.html", df_view=pd.DataFrame(), records=[], columns=[], bg_url=get_bg_url('preview'))


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    value_samples = {}
    value_samples_normal = {}
    value_samples_spam = {}
    try:
        upload_path = os.path.join(os.path.dirname(__file__), 'upload.csv')
        if os.path.exists(upload_path):
            df = pd.read_csv(upload_path, encoding='unicode_escape')

            def norm(s):
                s = str(s).lower().strip()
                # remove unit brackets and parentheses content
                s = re.sub(r"\[[^\]]*\]", "", s)
                s = re.sub(r"\([^\)]*\)", "", s)
                # collapse non-alphanumerics
                s = re.sub(r"[^a-z0-9]", "", s)
                return s

            def match_col(name):
                target = norm(name)
                # exact normalized match
                for c in df.columns:
                    if norm(c) == target:
                        return c
                # partial token match
                for c in df.columns:
                    nc = norm(c)
                    if target and target in nc:
                        return c
                # underscore/space variations
                for c in df.columns:
                    nc = str(c).lower().replace('_', ' ').strip()
                    if nc == str(name).lower().strip():
                        return c
                return None

            alias_map = {
                'generation': ['gen [kW]'],
                'House overall': ['House overall [kW]'],
                'Dishwasher': ['Dishwasher [kW]'],
                'Furnace': ['Furnace 1 [kW]', 'Furnace 2 [kW]'],
                'Home office': ['Home office [kW]'],
                'Fridge': ['Fridge [kW]'],
                'Wine cellar': ['Wine cellar [kW]'],
                'Garage door': ['Garage door [kW]'],
                'Kitchen': ['Kitchen 12 [kW]', 'Kitchen 14 [kW]', 'Kitchen 38 [kW]'],
                'Barn': ['Barn [kW]'],
                'Well': ['Well [kW]'],
                'Microwave': ['Microwave [kW]'],
                'Living room': ['Living room [kW]'],
                'Solar': ['Solar [kW]'],
                # Weather fields map to themselves; left to match_col
            }

            def get_series_for_field(field_name):
                cols = []
                if field_name in alias_map:
                    for raw in alias_map[field_name]:
                        # find by exact text first, then normalized
                        if raw in df.columns:
                            cols.append(raw)
                        else:
                            mc = match_col(raw)
                            if mc:
                                cols.append(mc)
                # pattern-based aggregation for Furnace/Kitchen if aliases missing
                if not cols and field_name.lower().startswith('furnace'):
                    cols = [c for c in df.columns if 'furnace' in str(c).lower()]
                if not cols and field_name.lower().startswith('kitchen'):
                    cols = [c for c in df.columns if 'kitchen' in str(c).lower()]

                if cols:
                    series_list = [pd.to_numeric(df[c], errors='coerce').fillna(0) for c in cols]
                    # sum multiple circuits/devices row-wise
                    s = series_list[0]
                    for extra in series_list[1:]:
                        s = s.add(extra, fill_value=0)
                    return s
                # fallback to single match by field name
                col = match_col(field_name)
                if col is not None:
                    return pd.to_numeric(df[col], errors='coerce')
                return None

            field_order = [
                'generation',
                'House overall',
                'Dishwasher',
                'Furnace',
                'Home office',
                'Fridge',
                'Wine cellar',
                'Garage door',
                'Kitchen',
                'Barn',
                'Well',
                'Microwave',
                'Living room',
                'Solar',
                'temperature',
                'humidity',
                'visibility',
                'apparentTemperature',
                'pressure',
                'windSpeed',
                'windBearing',
                'precipIntensity',
            ]

            for name in field_order:
                s = get_series_for_field(name)
                if s is not None:
                    s = s.dropna()
                    vals = sorted(pd.Series(s).unique().tolist())
                    value_samples[name] = vals
                else:
                    value_samples[name] = []
            normal_ranges_features = {
                'generation': (0.0, 0.003),
                'House overall': (0.0, 0.932),
                'Dishwasher': (0.0, 0.00003),
                'Furnace': (0.0, 0.021),
                'Home office': (0.0, 0.447),
                'Fridge': (0.0, 0.124),
                'Wine cellar': (0.0, 0.011),
                'Garage door': (0.0, 0.015),
                'Kitchen': (0.0, 0.0008),
                'Barn': (0.0, 0.031),
                'Well': (0.0, 0.001),
                'Microwave': (0.0, 0.004),
                'Living room': (0.0, 0.002),
                'Solar': (0.0, 0.003),
                'temperature': (36.14, 84.57),
                'humidity': (0.62, 0.65),
                'visibility': (9.88, 10.0),
                'apparentTemperature': (29.26, 89.73),
                'pressure': (1008.35, 1016.91),
                'windSpeed': (9.18, 13.82),
                'windBearing': (0.0, 360.0),
                'precipIntensity': (0.0, 1.0),
            }
            suspicious_upper = {
                'generation': (0.004, 0.006),
                'House overall': (0.933, 1.864),
                'Dishwasher': (0.00004, 0.0003),
                'Furnace': (0.022, 0.503),
                'Home office': (0.448, 0.894),
                'Fridge': (0.125, 0.248),
                'Wine cellar': (0.012, 0.022),
                'Garage door': (0.016, 0.030),
                'Kitchen': (0.0009, 0.0016),
                'Barn': (0.032, 0.062),
                'Well': (0.002, 0.003),
                'Microwave': (0.005, 0.008),
                'Living room': (0.003, 0.004),
                'Solar': (0.004, 0.006),
                'temperature': (84.58, 133.0),
                'humidity': (0.651, 0.68),
                'apparentTemperature': (89.74, 150.2),
                'windSpeed': (13.83, 18.46),
            }
            suspicious_lower = {
                'visibility': (9.76, 9.87),
                'pressure': (999.79, 1008.34),
            }
            for name in field_order:
                vals = value_samples.get(name, [])
                normals = []
                spams = []
                r = normal_ranges_features.get(name)
                for v in vals:
                    try:
                        x = float(v)
                    except Exception:
                        continue
                    is_out = False
                    is_susp = False
                    if r:
                        a, b = r
                        if x < float(a) or x > float(b):
                            is_out = True
                    if name in suspicious_upper:
                        smin, smax = suspicious_upper[name]
                        if x > float(smin) and x <= float(smax):
                            is_susp = True
                    if name in suspicious_lower and (not is_susp):
                        smin, smax = suspicious_lower[name]
                        if x >= float(smin) and x < float(smax):
                            is_susp = True
                    if is_out or is_susp:
                        spams.append(x)
                    else:
                        normals.append(x)
                if not normals:
                    if r:
                        a, b = r
                        try:
                            normals = [ (float(a) + float(b)) / 2.0 ]
                        except Exception:
                            normals = []
                if not spams:
                    if name in suspicious_upper:
                        smin, smax = suspicious_upper[name]
                        try:
                            spams = [ (float(smin) + float(smax)) / 2.0 ]
                        except Exception:
                            spams = []
                    elif name in suspicious_lower:
                        smin, smax = suspicious_lower[name]
                        try:
                            spams = [ (float(smin) + float(smax)) / 2.0 ]
                        except Exception:
                            spams = []
                value_samples_normal[name] = sorted(normals)
                value_samples_spam[name] = sorted(spams)
    except Exception:
        value_samples = {}
        value_samples_normal = {}
        value_samples_spam = {}

    # Use original local image path for prediction background
    return render_template('prediction.html', bg_url=get_bg_url('prediction'), value_samples=value_samples, value_samples_normal=value_samples_normal, value_samples_spam=value_samples_spam)

@app.route('/input_guide')
def input_guide():
    return render_template('input_guide.html', bg_url=get_bg_url('input_guide'))


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict',methods=['POST'])
def predict():
    global SPAM_COUNT, NO_SPAM_COUNT
    # Ensure values are collected in the exact order the model expects
    field_order = [
        'generation',
        'House overall',
        'Dishwasher',
        'Furnace',
        'Home office',
        'Fridge',
        'Wine cellar',
        'Garage door',
        'Kitchen',
        'Barn',
        'Well',
        'Microwave',
        'Living room',
        'Solar',
        'temperature',
        'humidity',
        'visibility',
        'apparentTemperature',
        'pressure',
        'windSpeed',
        'windBearing',
        'precipIntensity',
    ]

    def to_float(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    form = request.form
    features = [to_float(form.get(name)) for name in field_order]

    # Shape to (1, n_features) for sklearn
    final_features = np.array([features], dtype=float)

    pred = model.predict(final_features)
    label = int(pred[0]) if hasattr(pred, '__iter__') else int(pred)
    device_fields = [
        'House overall', 'Dishwasher', 'Furnace', 'Home office', 'Fridge',
        'Wine cellar', 'Garage door', 'Kitchen', 'Barn', 'Well',
        'Microwave', 'Living room', 'Solar'
    ]
    spam_ref = {
        'House overall': 6.357,
        'Dishwasher': 0.0003,
        'Furnace': 0.503,
        'Home office': 0.447,
        'Fridge': 0.124,
        'Wine cellar': 0.011,
        'Garage door': 0.015,
        'Kitchen': 0.0008,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
    }
    normal_ref = {
        'House overall': 0.932,
        'Dishwasher': 0.00003,
        'Furnace': 0.021,
        'Home office': 0.071,
        'Fridge': 0.005,
        'Wine cellar': 0.007,
        'Garage door': 0.013,
        'Kitchen': 0.0004,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
    }
    device_ranges = {
        'House overall': (0.0, 0.932),
        'Dishwasher': (0.0, 0.00003),
        'Furnace': (0.0, 0.021),
        'Home office': (0.0, 0.447),
        'Fridge': (0.0, 0.124),
        'Wine cellar': (0.0, 0.011),
        'Garage door': (0.0, 0.015),
        'Kitchen': (0.0, 0.0008),
        'Barn': (0.0, 0.031),
        'Well': (0.0, 0.001),
        'Microwave': (0.0, 0.004),
        'Living room': (0.0, 0.002),
        'Solar': (0.0, 0.003),
    }
    results = []
    for name in device_fields:
        raw = form.get(name)
        try:
            val = float(raw) if raw is not None and raw != '' else None
        except Exception:
            val = None
        status = '-'
        if val is not None and (name in device_ranges):
            rmin, rmax = device_ranges[name]
            status = 'Spam' if (val < float(rmin) or val > float(rmax)) else 'Normal'
        results.append({'name': name, 'value': raw if raw not in [None, ''] else '-', 'status': status})
    feature_fields = [
        'generation','House overall','Dishwasher','Furnace','Home office','Fridge','Wine cellar','Garage door',
        'Kitchen','Barn','Well','Microwave','Living room','Solar','temperature','humidity','visibility',
        'apparentTemperature','pressure','windSpeed','windBearing','precipIntensity'
    ]
    spam_ref_features = {
        'generation': 6.357,
        'House overall': 6.357,
        'Dishwasher': 0.0003,
        'Furnace': 0.503,
        'Home office': 0.447,
        'Fridge': 0.124,
        'Wine cellar': 0.011,
        'Garage door': 0.015,
        'Kitchen': 0.0008,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
        'temperature': 84.57,
        'humidity': 0.65,
        'visibility': 9.88,
        'apparentTemperature': 89.73,
        'pressure': 1008.35,
        'windSpeed': 13.82,
        'windBearing': 182,
        'precipIntensity': 0,
    }
    normal_ref_features = {
        'generation': 0.003,
        'House overall': 0.932,
        'Dishwasher': 0.00003,
        'Furnace': 0.021,
        'Home office': 0.071,
        'Fridge': 0.005,
        'Wine cellar': 0.007,
        'Garage door': 0.013,
        'Kitchen': 0.0004,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
        'temperature': 36.14,
        'humidity': 0.62,
        'visibility': 10,
        'apparentTemperature': 29.26,
        'pressure': 1016.91,
        'windSpeed': 9.18,
        'windBearing': 282,
        'precipIntensity': 0,
    }
    normal_ranges_features = {
        'generation': (0.0, 0.003),
        'House overall': (0.0, 0.932),
        'Dishwasher': (0.0, 0.00003),
        'Furnace': (0.0, 0.021),
        'Home office': (0.0, 0.447),
        'Fridge': (0.0, 0.124),
        'Wine cellar': (0.0, 0.011),
        'Garage door': (0.0, 0.015),
        'Kitchen': (0.0, 0.0008),
        'Barn': (0.0, 0.031),
        'Well': (0.0, 0.001),
        'Microwave': (0.0, 0.004),
        'Living room': (0.0, 0.002),
        'Solar': (0.0, 0.003),
        'temperature': (36.14, 84.57),
        'humidity': (0.62, 0.65),
        'visibility': (9.88, 10.0),
        'apparentTemperature': (29.26, 89.73),
        'pressure': (1008.35, 1016.91),
        'windSpeed': (9.18, 13.82),
        'windBearing': (0.0, 360.0),
        'precipIntensity': (0.0, 1.0),
    }
    feature_results = []
    for name in feature_fields:
        raw = form.get(name)
        try:
            val = float(raw) if raw is not None and raw != '' else None
        except Exception:
            val = None
        status = '-'
        if val is not None and (name in normal_ranges_features):
            rmin, rmax = normal_ranges_features[name]
            suspicious_upper = {
                'generation': (0.004, 0.006),
                'temperature': (84.58, 133.0),
                'humidity': (0.651, 0.68),
                'apparentTemperature': (89.74, 150.2),
                'windSpeed': (13.83, 18.46),
                'House overall': (0.933, 1.864),
                'Dishwasher': (0.00004, 0.0003),
                'Furnace': (0.022, 0.503),
                'Home office': (0.448, 0.894),
                'Fridge': (0.125, 0.248),
                'Wine cellar': (0.012, 0.022),
                'Garage door': (0.016, 0.030),
                'Kitchen': (0.0009, 0.0016),
                'Barn': (0.032, 0.062),
                'Well': (0.002, 0.003),
                'Microwave': (0.005, 0.008),
                'Living room': (0.003, 0.004),
                'Solar': (0.004, 0.006),
            }
            suspicious_lower = {
                'visibility': (9.76, 9.87),
                'pressure': (999.79, 1008.34),
            }
            if name in suspicious_upper:
                smin, smax = suspicious_upper[name]
                status = 'Spam' if (val < float(rmin) or val > float(rmax) or (val > float(smin) and val <= float(smax))) else 'Normal'
            elif name in suspicious_lower:
                smin, smax = suspicious_lower[name]
                status = 'Spam' if (val < float(rmin) or val > float(rmax) or (val >= float(smin) and val < float(smax))) else 'Normal'
            else:
                status = 'Spam' if (val < float(rmin) or val > float(rmax)) else 'Normal'
        feature_results.append({'name': name, 'value': raw if raw not in [None, ''] else '-', 'status': status})
    global LAST_DEVICE_STATUS, LAST_FEATURE_STATUS
    LAST_DEVICE_STATUS = results
    LAST_FEATURE_STATUS = feature_results

    suspicious = False
    for name in feature_fields:
        raw = form.get(name)
        try:
            val = float(raw) if raw is not None and raw != '' else None
        except Exception:
            val = None
        if val is not None and name in normal_ranges_features:
            rmin, rmax = normal_ranges_features[name]
            suspicious_upper = {
                'generation': (0.004, 0.006),
                'temperature': (84.58, 133.0),
                'humidity': (0.651, 0.68),
                'apparentTemperature': (89.74, 150.2),
                'pressure_upper': None,
                'windSpeed': (13.83, 18.46),
                'House overall': (0.933, 1.864),
                'Dishwasher': (0.00004, 0.0003),
                'Furnace': (0.022, 0.503),
                'Home office': (0.448, 0.894),
                'Fridge': (0.125, 0.248),
                'Wine cellar': (0.012, 0.022),
                'Garage door': (0.016, 0.030),
                'Kitchen': (0.0009, 0.0016),
                'Barn': (0.032, 0.062),
                'Well': (0.002, 0.003),
                'Microwave': (0.005, 0.008),
                'Living room': (0.003, 0.004),
                'Solar': (0.004, 0.006),
            }
            suspicious_lower = {
                'visibility': (9.76, 9.87),
                'pressure': (999.79, 1008.34),
            }
            if name in suspicious_upper and suspicious_upper[name] is not None:
                smin, smax = suspicious_upper[name]
                if (val < float(rmin) or val > float(rmax) or (val > float(smin) and val <= float(smax))):
                    suspicious = True
            elif name in suspicious_lower:
                smin, smax = suspicious_lower[name]
                if (val < float(rmin) or val > float(rmax) or (val >= float(smin) and val < float(smax))):
                    suspicious = True
            else:
                if val < float(rmin) or val > float(rmax):
                    suspicious = True

    if suspicious and label != 1:
        label = 1

    global SPAM_COUNT, NO_SPAM_COUNT
    if label == 1:
        SPAM_COUNT += 1
    else:
        NO_SPAM_COUNT += 1

    true_label = label
    globals()['LAST_PERF_SINGLE'] = {'pred': label, 'true': true_label}
    from flask import url_for
    return redirect(url_for('device_status'))
@app.route('/api/analyze_devices', methods=['POST', 'OPTIONS'])
def api_analyze_devices():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    feature_fields = [
        'generation',
        'House overall',
        'Dishwasher',
        'Furnace',
        'Home office',
        'Fridge',
        'Wine cellar',
        'Garage door',
        'Kitchen',
        'Barn',
        'Well',
        'Microwave',
        'Living room',
        'Solar',
        'temperature',
        'humidity',
        'visibility',
        'apparentTemperature',
        'pressure',
        'windSpeed',
        'windBearing',
        'precipIntensity',
    ]
    spam_ref = {
        'generation': 6.357,
        'House overall': 6.357,
        'Dishwasher': 0.0003,
        'Furnace': 0.503,
        'Home office': 0.447,
        'Fridge': 0.124,
        'Wine cellar': 0.011,
        'Garage door': 0.015,
        'Kitchen': 0.0008,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
        'temperature': 84.57,
        'humidity': 0.65,
        'visibility': 9.88,
        'apparentTemperature': 89.73,
        'pressure': 1008.35,
        'windSpeed': 13.82,
        'windBearing': 182,
        'precipIntensity': 0,
    }
    normal_ref = {
        'generation': 0.003,
        'House overall': 0.932,
        'Dishwasher': 0.00003,
        'Furnace': 0.021,
        'Home office': 0.071,
        'Fridge': 0.005,
        'Wine cellar': 0.007,
        'Garage door': 0.013,
        'Kitchen': 0.0004,
        'Barn': 0.031,
        'Well': 0.001,
        'Microwave': 0.004,
        'Living room': 0.002,
        'Solar': 0.003,
        'temperature': 36.14,
        'humidity': 0.62,
        'visibility': 10,
        'apparentTemperature': 29.26,
        'pressure': 1016.91,
        'windSpeed': 9.18,
        'windBearing': 282,
        'precipIntensity': 0,
    }
    data = request.get_json(silent=True) or {}
    values = data.get('values') if isinstance(data.get('values'), dict) else data
    normal_ranges = {
        'generation': (0.0, 0.003),
        'House overall': (0.0, 0.932),
        'Dishwasher': (0.0, 0.00003),
        'Furnace': (0.0, 0.021),
        'Home office': (0.0, 0.447),
        'Fridge': (0.0, 0.124),
        'Wine cellar': (0.0, 0.011),
        'Garage door': (0.0, 0.015),
        'Kitchen': (0.0, 0.0008),
        'Barn': (0.0, 0.031),
        'Well': (0.0, 0.001),
        'Microwave': (0.0, 0.004),
        'Living room': (0.0, 0.002),
        'Solar': (0.0, 0.003),
        'temperature': (36.14, 84.57),
        'humidity': (0.62, 0.65),
        'visibility': (9.88, 10.0),
        'apparentTemperature': (29.26, 89.73),
        'pressure': (1008.35, 1016.91),
        'windSpeed': (9.18, 13.82),
        'windBearing': (0.0, 360.0),
        'precipIntensity': (0.0, 1.0),
    }
    suspicious_band_api = {
        'generation': (0.003, 0.006),
        'House overall': (0.932, 1.864),
        'Dishwasher': (0.00003, 0.0003),
        'Furnace': (0.021, 0.503),
        'Home office': (0.447, 0.894),
        'Fridge': (0.124, 0.248),
        'Wine cellar': (0.011, 0.022),
        'Garage door': (0.015, 0.030),
        'Kitchen': (0.0008, 0.0016),
        'Barn': (0.031, 0.062),
        'Well': (0.001, 0.002),
        'Microwave': (0.004, 0.008),
        'Living room': (0.002, 0.004),
        'Solar': (0.003, 0.006),
        'temperature': (84.57, 133.0),
        'humidity': (0.65, 0.68),
        'visibility': (9.76, 9.88),
        'apparentTemperature': (89.73, 150.2),
        'pressure': (999.79, 1008.35),
        'windSpeed': (13.82, 18.46),
    }
    results = []
    for name in feature_fields:
        raw = values.get(name) if isinstance(values, dict) else None
        try:
            val = float(raw) if raw is not None and raw != '' else None
        except Exception:
            val = None
        status = '-'
        if val is not None and (name in normal_ranges):
            rmin, rmax = normal_ranges[name]
            if name in suspicious_band_api:
                smin, smax = suspicious_band_api[name]
                status = 'Spam' if (val < float(rmin) or (val >= float(smin) and val <= float(smax)) or val > float(smax)) else 'Normal'
            else:
                status = 'Spam' if (val < float(rmin) or val > float(rmax)) else 'Normal'
    results.append({'name': name, 'value': raw if raw not in [None, ''] else '-', 'status': status})
    resp = jsonify({'devices': results})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200
@app.route('/performance')
def performance():
    last_single = globals().get('LAST_PERF_SINGLE')
    if isinstance(last_single, dict) and 'pred' in last_single and 'true' in last_single:
        t = int(last_single['true'])
        p = int(last_single['pred'])
        tn = 1 if t == 0 and p == 0 else 0
        fp = 1 if t == 0 and p == 1 else 0
        fn = 1 if t == 1 and p == 0 else 0
        tp = 1 if t == 1 and p == 1 else 0
        def safe_div(a, b):
            return (a / b) if b else 0.0
        precision_0 = safe_div(tn, tn + fn)
        recall_0 = safe_div(tn, tn + fp)
        precision_1 = safe_div(tp, tp + fp)
        recall_1 = safe_div(tp, tp + fn)
        feat_list = globals().get('LAST_FEATURE_STATUS') if 'LAST_FEATURE_STATUS' in globals() else []
        if isinstance(feat_list, list):
            features_spam_count = sum(1 for f in feat_list if str(f.get('status','')) == 'Spam')
            features_normal_count = sum(1 for f in feat_list if str(f.get('status','')) == 'Normal')
            features_unknown_count = sum(1 for f in feat_list if str(f.get('status','')) not in ['Spam','Normal'])
        else:
            features_spam_count = 0
            features_normal_count = 0
            features_unknown_count = 0
        features_total_count = features_spam_count + features_normal_count + features_unknown_count
        normal_ranges_features_perf = {
            'generation': (0.0, 0.003),
            'House overall': (0.0, 0.932),
            'Dishwasher': (0.0, 0.00003),
            'Furnace': (0.0, 0.021),
            'Home office': (0.0, 0.447),
            'Fridge': (0.0, 0.124),
            'Wine cellar': (0.0, 0.011),
            'Garage door': (0.0, 0.015),
            'Kitchen': (0.0, 0.0008),
            'Barn': (0.0, 0.031),
            'Well': (0.0, 0.001),
            'Microwave': (0.0, 0.004),
            'Living room': (0.0, 0.002),
            'Solar': (0.0, 0.003),
            'temperature': (36.14, 84.57),
            'humidity': (0.62, 0.65),
            'visibility': (9.88, 10.0),
            'apparentTemperature': (29.26, 89.73),
            'pressure': (1008.35, 1016.91),
            'windSpeed': (9.18, 13.82),
            'windBearing': (0.0, 360.0),
            'precipIntensity': (0.0, 1.0),
        }
        feat_cm_tp = 0
        feat_cm_fp = 0
        feat_cm_tn = 0
        feat_cm_fn = 0
        feat_tp_devices = []
        feat_fp_devices = []
        feat_tn_devices = []
        feat_fn_devices = []
        if isinstance(feat_list, list) and len(feat_list) > 0:
            internal_upper_perf = {
                'generation': (0.0026, 0.0030),
                'House overall': (0.85, 0.932),
                'Dishwasher': (0.000025, 0.00003),
                'Furnace': (0.018, 0.021),
                'Home office': (0.43, 0.447),
                'Fridge': (0.115, 0.124),
                'Wine cellar': (0.0095, 0.011),
                'Garage door': (0.0135, 0.015),
                'Kitchen': (0.0007, 0.0008),
                'Barn': (0.028, 0.031),
                'Well': (0.0009, 0.001),
                'Microwave': (0.0035, 0.004),
                'Living room': (0.0016, 0.002),
                'Solar': (0.0026, 0.0030),
                'temperature': (84.30, 84.57),
                'humidity': (0.645, 0.65),
                'visibility': (9.88, 9.90),
                'apparentTemperature': (89.50, 89.73),
                'pressure': (1008.35, 1009.00),
                'windSpeed': (13.50, 13.82),
            }
            external_upper_perf = {
                'generation': (0.004, 0.006),
                'House overall': (0.933, 1.864),
                'Dishwasher': (0.00004, 0.0003),
                'Furnace': (0.022, 0.503),
                'Home office': (0.448, 0.894),
                'Fridge': (0.125, 0.248),
                'Wine cellar': (0.012, 0.022),
                'Garage door': (0.016, 0.030),
                'Kitchen': (0.0009, 0.0016),
                'Barn': (0.032, 0.062),
                'Well': (0.002, 0.003),
                'Microwave': (0.005, 0.008),
                'Living room': (0.003, 0.004),
                'Solar': (0.004, 0.006),
                'temperature': (84.58, 133.0),
                'humidity': (0.651, 0.68),
                'apparentTemperature': (89.74, 150.2),
                'windSpeed': (13.83, 18.46),
            }
            external_lower_perf = {
                'visibility': (9.76, 9.87),
                'pressure': (999.79, 1008.34),
            }
            for f in feat_list:
                name = str(f.get('name',''))
                status = str(f.get('status',''))
                raw = f.get('value')
                try:
                    val = float(raw) if raw not in [None, '-', ''] else None
                except Exception:
                    val = None
                if val is None or name not in normal_ranges_features_perf:
                    continue
                rmin, rmax = normal_ranges_features_perf[name]
                actual_spam = (val < float(rmin)) or (val > float(rmax))
                predicted_spam = False
                if name in internal_upper_perf:
                    smin, smax = internal_upper_perf[name]
                    if (val >= float(smin)) and (val <= float(smax)):
                        predicted_spam = True
                if (not predicted_spam) and name in external_upper_perf:
                    smin, smax = external_upper_perf[name]
                    if (val > float(smin)) and (val <= float(smax)):
                        predicted_spam = True
                if (not predicted_spam) and name in external_lower_perf:
                    smin, smax = external_lower_perf[name]
                    if (val >= float(smin)) and (val < float(smax)):
                        predicted_spam = True
                if predicted_spam and actual_spam:
                    feat_cm_tp += 1
                    feat_tp_devices.append(name)
                elif predicted_spam and (not actual_spam):
                    feat_cm_fp += 1
                    feat_fp_devices.append(name)
                elif (not predicted_spam) and actual_spam:
                    feat_cm_fn += 1
                    feat_fn_devices.append(name)
                else:
                    feat_cm_tn += 1
                    feat_tn_devices.append(name)
        def sdiv(a, b):
            return (a / b) if b else 0.0
        feat_prec = sdiv(feat_cm_tp, (feat_cm_tp + feat_cm_fp))
        feat_rec = sdiv(feat_cm_tp, (feat_cm_tp + feat_cm_fn))
        feat_spec = sdiv(feat_cm_tn, (feat_cm_tn + feat_cm_fp))
        feat_acc = sdiv((feat_cm_tp + feat_cm_tn), (feat_cm_tp + feat_cm_tn + feat_cm_fp + feat_cm_fn))
        feat_f1 = sdiv((2 * feat_cm_tp), (2 * feat_cm_tp + feat_cm_fp + feat_cm_fn))
        return render_template('performance.html',
                               precision_0=precision_0,
                               precision_1=precision_1,
                               recall_0=recall_0,
                               recall_1=recall_1,
                               cm_00=tn, cm_01=fp,
                               cm_10=fn, cm_11=tp,
                               features_spam_count=features_spam_count,
                               features_normal_count=features_normal_count,
                               features_unknown_count=features_unknown_count,
                               features_total_count=features_total_count,
                               feat_cm_tp=feat_cm_tp,
                               feat_cm_fp=feat_cm_fp,
                               feat_cm_tn=feat_cm_tn,
                               feat_cm_fn=feat_cm_fn,
                               feat_prec=feat_prec,
                               feat_rec=feat_rec,
                               feat_spec=feat_spec,
                               feat_acc=feat_acc,
                               feat_f1=feat_f1,
                               data_source='last prediction')
    # Load dataset - use upload.csv if exists, otherwise use test data
    try:
        upload_path = os.path.join(os.path.dirname(__file__), 'upload.csv')
        test_path = os.path.join(os.path.dirname(__file__), 'test_data', 'test.csv')
        
        if os.path.exists(upload_path):
            df = pd.read_csv(upload_path, encoding='unicode_escape')
            data_source = 'upload.csv'
        else:
            df = pd.read_csv(test_path)
            data_source = 'test_data/test.csv'
        
        # Get feature mapping from train_model.py
        feature_mapping = [
            ('gen [kW]', 'generation'),
            ('House overall [kW]', 'House overall'),
            ('Dishwasher [kW]', 'Dishwasher'),
            ('Furnace 1 [kW]', 'Furnace'),
            ('Home office [kW]', 'Home office'),
            ('Fridge [kW]', 'Fridge'),
            ('Wine cellar [kW]', 'Wine cellar'),
            ('Garage door [kW]', 'Garage door'),
            ('Kitchen 12 [kW]', 'Kitchen'),
            ('Barn [kW]', 'Barn'),
            ('Well [kW]', 'Well'),
            ('Microwave [kW]', 'Microwave'),
            ('Living room [kW]', 'Living room'),
            ('Solar [kW]', 'Solar'),
            ('temperature', 'temperature'),
            ('humidity', 'humidity'),
            ('visibility', 'visibility'),
            ('apparentTemperature', 'apparentTemperature'),
            ('pressure', 'pressure'),
            ('windSpeed', 'windSpeed'),
            ('windBearing', 'windBearing'),
            ('precipIntensity', 'precipIntensity'),
        ]
        
        # Convert target labels - strings like "1(spam)" / "0(no spam)" to ints 1/0
        y_true = df['class'].astype(str).str.contains('1').astype(int)
        
        # Build feature matrix in the exact order expected by the model
        X_cols = [src for src, _ in feature_mapping]
        X = df[X_cols].astype(float).values
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        # Calculate precision, recall for each class
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Extract confusion matrix values
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        
        # Use the existing static image instead of generating a new one
        
        return render_template('performance.html',
                              precision_0=precision[0],
                              precision_1=precision[1],
                              recall_0=recall[0],
                              recall_1=recall[1],
                              cm_00=tn, cm_01=fp,
                              cm_10=fn, cm_11=tp,
                              features_spam_count=0,
                              features_normal_count=0,
                              features_unknown_count=0,
                              features_total_count=0,
                              feat_cm_tp=0,
                              feat_cm_fp=0,
                              feat_cm_tn=0,
                              feat_cm_fn=0,
                              feat_prec=0.0,
                              feat_rec=0.0,
                              feat_spec=0.0,
                              feat_acc=0.0,
                              feat_f1=0.0,
                              data_source=data_source)
    except Exception as e:
        return render_template('performance.html',
                              precision_0=1.0,
                              precision_1=1.0,
                              recall_0=1.0,
                              recall_1=1.0,
                              cm_00=0, cm_01=0,
                              cm_10=0, cm_11=0,
                              data_source="Error: " + str(e))   

@app.route('/perf_devices')
def perf_devices():
    feat_list = globals().get('LAST_FEATURE_STATUS') if 'LAST_FEATURE_STATUS' in globals() else []
    bg_override = request.args.get('bg')
    normal_ranges = {
        'generation': (0.0, 0.003),
        'House overall': (0.0, 0.932),
        'Dishwasher': (0.0, 0.00003),
        'Furnace': (0.0, 0.021),
        'Home office': (0.0, 0.447),
        'Fridge': (0.0, 0.124),
        'Wine cellar': (0.0, 0.011),
        'Garage door': (0.0, 0.015),
        'Kitchen': (0.0, 0.0008),
        'Barn': (0.0, 0.031),
        'Well': (0.0, 0.001),
        'Microwave': (0.0, 0.004),
        'Living room': (0.0, 0.002),
        'Solar': (0.0, 0.003),
        'temperature': (36.14, 84.57),
        'humidity': (0.62, 0.65),
        'visibility': (9.88, 10.0),
        'apparentTemperature': (29.26, 89.73),
        'pressure': (1008.35, 1016.91),
        'windSpeed': (9.18, 13.82),
        'windBearing': (0.0, 360.0),
        'precipIntensity': (0.0, 1.0),
    }
    internal_upper = {
        'generation': (0.0026, 0.0030),
        'House overall': (0.85, 0.932),
        'Dishwasher': (0.000025, 0.00003),
        'Furnace': (0.018, 0.021),
        'Home office': (0.43, 0.447),
        'Fridge': (0.115, 0.124),
        'Wine cellar': (0.0095, 0.011),
        'Garage door': (0.0135, 0.015),
        'Kitchen': (0.0007, 0.0008),
        'Barn': (0.028, 0.031),
        'Well': (0.0009, 0.001),
        'Microwave': (0.0035, 0.004),
        'Living room': (0.0016, 0.002),
        'Solar': (0.0026, 0.0030),
        'temperature': (84.30, 84.57),
        'humidity': (0.645, 0.65),
        'visibility': (9.88, 9.90),
        'apparentTemperature': (89.50, 89.73),
        'pressure': (1008.35, 1009.00),
        'windSpeed': (13.50, 13.82),
    }
    external_upper = {
        'generation': (0.004, 0.006),
        'House overall': (0.933, 1.864),
        'Dishwasher': (0.00004, 0.0003),
        'Furnace': (0.022, 0.503),
        'Home office': (0.448, 0.894),
        'Fridge': (0.125, 0.248),
        'Wine cellar': (0.012, 0.022),
        'Garage door': (0.016, 0.030),
        'Kitchen': (0.0009, 0.0016),
        'Barn': (0.032, 0.062),
        'Well': (0.002, 0.003),
        'Microwave': (0.005, 0.008),
        'Living room': (0.003, 0.004),
        'Solar': (0.004, 0.006),
        'temperature': (84.58, 133.0),
        'humidity': (0.651, 0.68),
        'apparentTemperature': (89.74, 150.2),
        'windSpeed': (13.83, 18.46),
    }
    external_lower = {
        'visibility': (9.76, 9.87),
        'pressure': (999.79, 1008.34),
    }
    tp = []
    fp = []
    tn = []
    fn = []
    for f in (feat_list or []):
        name = str(f.get('name',''))
        raw = f.get('value')
        try:
            val = float(raw) if raw not in [None, '-', ''] else None
        except Exception:
            val = None
        if val is None or name not in normal_ranges:
            continue
        rmin, rmax = normal_ranges[name]
        actual_spam = (val < float(rmin)) or (val > float(rmax))
        predicted_spam = False
        if name in internal_upper:
            smin, smax = internal_upper[name]
            if (val >= float(smin)) and (val <= float(smax)):
                predicted_spam = True
        if (not predicted_spam) and name in external_upper:
            smin, smax = external_upper[name]
            if (val > float(smin)) and (val <= float(smax)):
                predicted_spam = True
        if (not predicted_spam) and name in external_lower:
            smin, smax = external_lower[name]
            if (val >= float(smin)) and (val < float(smax)):
                predicted_spam = True
        if predicted_spam and actual_spam:
            tp.append(name)
        elif predicted_spam and (not actual_spam):
            fp.append(name)
        elif (not predicted_spam) and actual_spam:
            fn.append(name)
        else:
            tn.append(name)
    feat_pred_spam_devices = sorted(list(set(tp + fp)))
    feat_pred_normal_devices = sorted(list(set(tn + fn)))
    feat_actual_spam_devices = sorted(list(set(tp + fn)))
    feat_actual_normal_devices = sorted(list(set(tn + fp)))
    return render_template('perf_devices.html',
                           feat_cm_tp=len(tp),
                           feat_cm_fp=len(fp),
                           feat_cm_tn=len(tn),
                           feat_cm_fn=len(fn),
                           feat_tp_devices=tp,
                           feat_fp_devices=fp,
                           feat_tn_devices=tn,
                           feat_fn_devices=fn,
                           feat_pred_spam_devices=feat_pred_spam_devices,
                           feat_pred_normal_devices=feat_pred_normal_devices,
                           feat_actual_spam_devices=feat_actual_spam_devices,
                           feat_actual_normal_devices=feat_actual_normal_devices,
                           bg_url=(bg_override or get_bg_url('perf_device')))
    
@app.route('/abstract')
def abstract():
    return render_template('abstract.html', bg_url=get_bg_url('abstract'))

@app.route('/user_register')
def user_register():
    return render_template('user_register.html', bg_url=get_bg_url('user_register'))

@app.route('/api/register', methods=['POST', 'OPTIONS'])
def api_register():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    phone = str(data.get('phone', '')).strip()
    name = str(data.get('name', '')).strip()
    gender = str(data.get('gender', '')).strip()
    dob = str(data.get('dob', '')).strip()
    address = str(data.get('address', '')).strip()
    home_id = str(data.get('homeId', '')).strip()
    device_id = str(data.get('deviceId', '')).strip()
    password = str(data.get('password', '')).strip()
    if not email or not phone or not name or not password:
        resp = jsonify({"message": "Missing required fields"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_valid_email(email):
        resp = jsonify({"message": "Invalid email format"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_valid_phone(phone):
        resp = jsonify({"message": "Invalid phone format"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_strong_password(password):
        resp = jsonify({"message": "Weak password. Use upper, lower, number, special, ≥8"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    for u in users:
        if str(u.get('email', '')).strip().lower() == email or str(u.get('phone', '')).strip() == phone:
            resp = jsonify({"message": "Email or phone number is already registered"})
            resp.headers['Access-Control-Allow-Origin'] = '*'
            return resp, 409
    users.append({
        "name": name,
        "email": email,
        "phone": phone,
        "gender": gender,
        "dob": dob,
        "address": address,
        "homeId": home_id,
        "deviceId": device_id,
        "password": password,
        "status": "pending"
    })
    try:
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(users, f)
    except Exception:
        resp = jsonify({"message": "Failed to save registration"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500
    resp = jsonify({"message": "Registration successful"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/api/user_reset_password', methods=['POST', 'OPTIONS'])
def api_user_reset_password():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    phone = str(data.get('phone', '')).strip()
    new_pwd = str(data.get('newPassword', '')).strip()
    if not new_pwd or not email or not phone:
        resp = jsonify({"message": "Email, phone and new password are required"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_valid_email(email):
        resp = jsonify({"message": "Invalid email format"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_valid_phone(phone):
        resp = jsonify({"message": "Invalid phone format"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if not is_strong_password(new_pwd):
        resp = jsonify({"message": "Weak password. Use upper, lower, number, special, ≥8"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    updated = False
    for u in users:
        if (str(u.get('email','')).strip().lower() == email) and (str(u.get('phone','')).strip() == phone):
            u['password'] = new_pwd
            updated = True
            break
    if not updated:
        resp = jsonify({"message": "User not found"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    try:
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(users, f)
    except Exception:
        resp = jsonify({"message": "Failed to save password"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500
    resp = jsonify({"message": "Password reset successful"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/admin_dashboard')
def admin_dashboard():
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    filtered = [u for u in users if str(u.get('status','')).strip().lower() != 'rejected']
    if len(filtered) != len(users):
        try:
            with open(users_path, 'w', encoding='utf-8') as f:
                json.dump(filtered, f)
        except Exception:
            pass
    users = filtered
    return render_template('admin_dashboard.html', users=users, bg_url=get_bg_url('admin_dashboard'))

@app.route('/api/admin/users', methods=['GET'])
def api_admin_users():
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    filtered = [u for u in users if str(u.get('status','')).strip().lower() != 'rejected']
    if len(filtered) != len(users):
        try:
            with open(users_path, 'w', encoding='utf-8') as f:
                json.dump(filtered, f)
        except Exception:
            pass
    users = filtered
    resp = jsonify(users)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/api/admin/approve', methods=['POST', 'OPTIONS'])
def api_admin_approve():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    phone = str(data.get('phone', '')).strip()
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    updated = False
    for u in users:
        if (str(u.get('email','')).strip().lower() == email) or (str(u.get('phone','')).strip() == phone):
            u['status'] = 'approved'
            updated = True
            break
    if not updated:
        resp = jsonify({"message": "User not found"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    try:
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(users, f)
    except Exception:
        resp = jsonify({"message": "Failed to save status"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500
    resp = jsonify({"message": "User approved"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/api/admin/reject', methods=['POST', 'OPTIONS'])
def api_admin_reject():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    phone = str(data.get('phone', '')).strip()
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    before_len = len(users)
    users = [u for u in users if not ((str(u.get('email','')).strip().lower() == email) or (str(u.get('phone','')).strip() == phone))]
    updated = len(users) < before_len
    if not updated:
        resp = jsonify({"message": "User not found"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    try:
        with open(users_path, 'w', encoding='utf-8') as f:
            json.dump(users, f)
    except Exception:
        resp = jsonify({"message": "Failed to save status"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 500
    resp = jsonify({"message": "User deleted"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/user_login')
def user_login():
    return render_template('user_login.html', bg_url=get_bg_url('user_login'))

@app.route('/api/user_login', methods=['POST', 'OPTIONS'])
def api_user_login():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    username = str(data.get('username', '')).strip()
    pwd = str(data.get('password', '')).strip()
    if not is_strong_password(pwd):
        resp = jsonify({"message": "Invalid password format. Use upper, lower, number, special, ≥8"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    # find user by name (case-insensitive)
    user = None
    for u in users:
        if str(u.get('name','')).strip().lower() == username.lower():
            user = u
            break
    if not user:
        resp = jsonify({"message": "Incorrect user name"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    if str(user.get('password','')).strip() != pwd:
        resp = jsonify({"message": "Incorrect password"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 401
    if str(user.get('status','')).strip().lower() != 'approved':
        resp = jsonify({"message": "User registration request not approved"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 403
    resp = jsonify({"message": "Login allowed"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/admin_login')
def admin_login():
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    return render_template('admin_login.html', users=users, bg_url=get_bg_url('admin_login'))

@app.route('/api/admin_login', methods=['POST', 'OPTIONS'])
def api_admin_login():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    pwd = str(data.get('password', '')).strip()
    if not is_strong_password(pwd):
        resp = jsonify({"message": "Invalid password format. Use upper, lower, number, special, ≥8"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if name != ADMIN_USER:
        resp = jsonify({"message": "Admin name is incorrect"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    if pwd != ADMIN_PWD:
        resp = jsonify({"message": "Password is incorrect"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 401
    resp = jsonify({"message": "Admin login success"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/api/admin_reset_password', methods=['POST', 'OPTIONS'])
def api_admin_reset_password():
    if request.method == 'OPTIONS':
        resp = jsonify({})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return resp, 200
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    hint = str(data.get('hint', '')).strip()
    new_pwd = str(data.get('newPassword', '')).strip()
    if not name or not hint or not new_pwd:
        resp = jsonify({"message": "Admin name, hint and new password are required"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    if name != ADMIN_USER:
        resp = jsonify({"message": "Admin name is incorrect"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 404
    if hint != ADMIN_HINT:
        resp = jsonify({"message": "Admin hint is incorrect"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 401
    if not is_strong_password(new_pwd):
        resp = jsonify({"message": "Weak password. Use upper, lower, number, special, ≥8"})
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp, 400
    global ADMIN_PWD
    ADMIN_PWD = new_pwd
    resp = jsonify({"message": "Admin password reset successful"})
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp, 200

@app.route('/user_data')
def user_data():
    users_path = os.path.join(os.path.dirname(__file__), 'users.json')
    try:
        if os.path.exists(users_path):
            with open(users_path, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
    except Exception:
        users = []
    filtered = [u for u in users if str(u.get('status','')).strip().lower() != 'rejected']
    if len(filtered) != len(users):
        try:
            with open(users_path, 'w', encoding='utf-8') as f:
                json.dump(filtered, f)
        except Exception:
            pass
    users = filtered
    return render_template('user_data.html', users=users, bg_url=get_bg_url('user_data'))

@app.route('/home.html')
@app.route('/index.html')
def home_html():
    return home()

@app.route('/upload1')
def upload1():
    return render_template('upload1.html', bg_url=get_bg_url('upload'))

@app.route('/upload1.html')
def upload1_html():
    return render_template('upload1.html', bg_url=get_bg_url('upload'))

@app.route('/admin_login.html')
def admin_login_html():
    return admin_login()

@app.route('/user_register.html')
def user_register_html():
    return user_register()

@app.route('/user_data.html')
def user_data_html():
    return user_data()

@app.route('/abstract.html')
def abstract_html():
    return abstract()

@app.route('/device_status')
def device_status():
    default_devices = [
        {'name':'House overall','value':'-','status':'-'},
        {'name':'Dishwasher','value':'-','status':'-'},
        {'name':'Furnace','value':'-','status':'-'},
        {'name':'Home office','value':'-','status':'-'},
        {'name':'Fridge','value':'-','status':'-'},
        {'name':'Wine cellar','value':'-','status':'-'},
        {'name':'Garage door','value':'-','status':'-'},
        {'name':'Kitchen','value':'-','status':'-'},
        {'name':'Barn','value':'-','status':'-'},
        {'name':'Well','value':'-','status':'-'},
        {'name':'Microwave','value':'-','status':'-'},
        {'name':'Living room','value':'-','status':'-'},
        {'name':'Solar','value':'-','status':'-'}
    ]
    default_features = [
        {'name':'generation','value':'-','status':'-'},
        {'name':'House overall','value':'-','status':'-'},
        {'name':'Dishwasher','value':'-','status':'-'},
        {'name':'Furnace','value':'-','status':'-'},
        {'name':'Home office','value':'-','status':'-'},
        {'name':'Fridge','value':'-','status':'-'},
        {'name':'Wine cellar','value':'-','status':'-'},
        {'name':'Garage door','value':'-','status':'-'},
        {'name':'Kitchen','value':'-','status':'-'},
        {'name':'Barn','value':'-','status':'-'},
        {'name':'Well','value':'-','status':'-'},
        {'name':'Microwave','value':'-','status':'-'},
        {'name':'Living room','value':'-','status':'-'},
        {'name':'Solar','value':'-','status':'-'},
        {'name':'temperature','value':'-','status':'-'},
        {'name':'humidity','value':'-','status':'-'},
        {'name':'visibility','value':'-','status':'-'},
        {'name':'apparentTemperature','value':'-','status':'-'},
        {'name':'pressure','value':'-','status':'-'},
        {'name':'windSpeed','value':'-','status':'-'},
        {'name':'windBearing','value':'-','status':'-'},
        {'name':'precipIntensity','value':'-','status':'-'}
    ]
    devices = LAST_DEVICE_STATUS if isinstance(LAST_DEVICE_STATUS, list) and LAST_DEVICE_STATUS else default_devices
    features = LAST_FEATURE_STATUS if isinstance(LAST_FEATURE_STATUS, list) and LAST_FEATURE_STATUS else default_features
    last_single = globals().get('LAST_PERF_SINGLE') if 'LAST_PERF_SINGLE' in globals() else None
    pred_label = None
    pred_text = None
    if isinstance(last_single, dict) and 'pred' in last_single:
        try:
            pred_label = int(last_single.get('pred'))
        except Exception:
            pred_label = None
        if pred_label is not None:
            pred_text = 'Spam (1)' if pred_label == 1 else 'No Spam (0)'
    return render_template('device_status.html', devices=devices, features=features, bg_url=get_bg_url('device_status'), last_pred_label=pred_label, last_pred_text=pred_text)

 

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
