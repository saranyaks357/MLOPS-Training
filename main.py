from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI(title='Iris RF Demo')
templates = Jinja2Templates(directory="templates")

try:
    model = joblib.load('artifacts/rf_iris.joblib')
except Exception:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    preds = model.predict(data)
    class_names = ['setosa', 'versicolor', 'virginica']
    prediction = class_names[preds[0]]
    return templates.TemplateResponse("form.html", {"request": request, "prediction": prediction})
