from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, SelectField, IntegerField
from wtforms.validators import DataRequired, Email

class MyForm(FlaskForm):
    EmploymentField = StringField("EmploymentField: ")
    EmploymentStatus = StringField("EmploymentStatus: ")
    Gender = SelectField("Gender: ", choices=[('male', 'male'), ('female', 'female')])
    LanguageAtHome = StringField("LanguageAtHome: ")
    JobWherePref = StringField("JobWherePref: ")
    SchoolDegree = StringField("SchoolDegree: ")
    Income = IntegerField("Income: ")
    submit = SubmitField("Submit")

class SaverForm(FlaskForm):
    submit = SubmitField("Submit")