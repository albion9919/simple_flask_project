from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from framework_bd import frame_wrapper, Preprocessor

app = Flask(__name__, static_url_path='/app/static')
# analyzer = aa.DispAnalyzer()
app.config.from_object('config')
file_name = "data\\2016-FCC-New-Coders-Survey-Data.csv"
factor_names = "CityPopulation EmploymentStatus Gender HasDebt JobPref JobWherePref MaritalStatus Income SchoolDegree"
sep = " "
numeric = "Income"
strx = "CityPopulation EmploymentStatus Gender HasDebt JobPref JobWherePref MaritalStatus SchoolDegree".split(" ")
fw = frame_wrapper(file_name=file_name,ind=factor_names,sep=sep, clear=True)

fw.to_numeric(numeric)
fw.to_strx(strx)

pre = Preprocessor()
from app import views, forms


# <br/>{% for plot_url in imgs %}
# <img src="data:image/png;base64, {{ plot_url }}">
# {% endfor %}