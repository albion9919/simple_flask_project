import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from app import app, fw, pre
from framework_bd import DispAnalyzer
from flask import render_template, redirect, url_for, Response
import pandas as pd
import pretty_html_table as pht
from app.forms import MyForm, SaverForm


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/AddObserve.html", methods=['GET', 'POST'])
def AddObserve():
    form = MyForm()
    if form.validate_on_submit():
        sx = pd.Series(
            [form.EmploymentField.data, form.EmploymentStatus.data, form.Gender.data, form.LanguageAtHome.data,
             form.JobWherePref.data,
             form.SchoolDegree.data, form.Income.data],
            index="EmploymentField, EmploymentStatus, Gender, LanguageAtHome, JobWherePref, SchoolDegree, Income".split(
                ", "))
        fw.append(sx)
        # здесь логика базы данных
        return redirect(url_for('contact'))
    return render_template('AddObserve.html', form=form)


@app.route('/ShowObserves.html')
def ShowObserves():
    return render_template('ShowObserves.html', tbl=pht.build_table(fw.get_df()[0:50], 'blue_light'))


@app.route("/SaveObserve.html", methods=['GET', 'POST'])
def route_SaveObserve():
    form = SaverForm()
    if form.validate_on_submit():
        # an.save()
        return render_template("SaveObserve.html", form=form, status="Successful saving")
    return render_template("SaveObserve.html", form=form, status="No changes")


#
# @app.route("/distplot-{number}.png")
# def plot_png(number):
#     """ renders the plot on the fly.
#     """
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     x_points = range(num_x_points)
#     axis.plot(x_points, [random.randint(1, 30) for x in x_points])
#
#     output = io.BytesIO()
#     FigureCanvasAgg(fig).print_png(output)
#     return Response(output.getvalue(), mimetype="image/png")


@app.route("/Analysis.html")
def Analysis():
    dependent_var = "Income"
    independent_var = ["SchoolDegree"]
    return analysis(dependent_var, independent_var)

@app.route("/GenderMaritalAnalysis.html")
def GenderMaritalAnalysis():
    dependent_var = "Income"
    independent_var = ["Gender", "MaritalStatus"]
    return analysis(dependent_var, independent_var)

def analysis(dependent_var, independent_var_multiply):
    analyzer = DispAnalyzer(fw.get_df(), dep=dependent_var, indep=independent_var_multiply)
    template = None
    if len (independent_var_multiply) == 1:
        template = analyzer.scenario_one_way()
    if len (independent_var_multiply) == 2:
        template = analyzer.scenario_two_ways()
    return template


@app.route("/GenderJobAnalysis.html")
def GenderJobAnalysis():
    dependent_var = "Income"
    independent_var_multiply = ["Gender", "JobPref"]
    return analysis(dependent_var, independent_var_multiply)
# analyzer.set_levels()
#
# img = StringIO()
# y = [1,2,3,4,5]
# x = [0,2,1,3,4]
#
# plt.plot(x,y)
# plt.savefig(img, format='png')
# plt.close()
# img.seek(0)
#
# plot_url = base64.b64encode(img.getvalue())


#
# @app.route('/Saver', methods=['GET', 'POST'])
# def saver():
#     form = SaverForm()
#     if form.validate_on_submit():
#         an.save()
#         return render_template("Saver.html", form=form, status="Successful saving")
#     return render_template("Saver.html", form=form, status="No changes")
#
#
# def anx(pair):
#     ct , ef ,res, r, p , g, sv = an.cross_analyze(pair[0],pair[1])
#     print(ct)
#     print(ef)
#     ef = pd.DataFrame(ef)
#     return render_template("Analysis.html",f1=pair[0],f2=pair[1],
#                            tbl=pht.build_table(ct, 'blue_light'),tbl2=pht.build_table(ef, 'blue_light'),analyz=res,result=r, p=p, g=g, sv=sv)
#
# # Gender, JobWherePref;
# @app.route('/GenderJobWherePref')
# def pair2():
#     pair = 'Gender','JobWherePref'
#     return anx(pair)
#
# @app.route('/AddObserve.html', methods=['GET', 'POST'])
# def contact():
#     form = MyForm()
#     if form.validate_on_submit():
#         sx = pd.Series(
#             [form.EmploymentField.data, form.EmploymentStatus.data, form.Gender.data, form.LanguageAtHome.data,
#              form.JobWherePref.data,
#              form.SchoolDegree.data, form.Income.data],
#             index="EmploymentField, EmploymentStatus, Gender, LanguageAtHome, JobWherePref, SchoolDegree, Income".split(
#                 ", "))
#         an.append(sx)
#         # здесь логика базы данных
#         return redirect(url_for('contact'))
#     return render_template('AddObserve.html', form=form)
#
# @app.route('/ShowObserves.html')
# def ShowObserves():
#     return render_template('ShowObserves.html', tbl=pht.build_table(an.df[0:50], 'blue_light'))
