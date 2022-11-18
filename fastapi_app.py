from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
import uvicorn
import pickle
import numpy as np

app = FastAPI (debug=True)

templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, 'y':''})

@app.post('/')
def ypred(request: Request, argument=Form(...)):
    argument = argument
    lis = [[argument]]
    lis = np.array(lis).reshape(-1, 1)

    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
        y_pred = model.predict(lis)

    return templates.TemplateResponse(
        "index.html",{
            "request": request,
            'y':{'x':argument, 'y_pred': y_pred[0]}
            }
            )


if __name__ == '__main__':
    uvicorn.run(app)