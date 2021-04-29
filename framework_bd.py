from flask import render_template, url_for
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.preprocessing as pre
import seaborn as sns
from sklearn.metrics import r2_score
import scipy.stats
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.stattools import (
    omni_normtest, durbin_watson)
import os
import pandas as pd
from math import exp
import pretty_html_table as pht
# pd.set_option('max_columns', 20)

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
class frame_wrapper:
    def __init__(self, file_name, ind=None, sep=None, clear=True):
        self.file = file_name
        self.df = pd.read_csv(file_name)
        if ind is not None:
            ind = ind.split(sep)
            self.df = self.df[ind]
        if clear:
            self.clear_data()

    def get_df(self):
        return self.df

    def save(self):
        self.df.to_csv(self.file, sep=',', encoding='utf-8')

    def append(self, s):
        self.df = self.df.append(s, ignore_index=True)

    def to_numeric(self, s):
        if isinstance(s, list):
            for i in s:
                self.df[i] = self.df[i].astype(float)
        else:
            self.df[s] = self.df[s].astype(float)

    def to_strx(self, s):
        if isinstance(s, list):
            for i in s:
                self.df[i] = self.df[i].astype(str)
        else:
            self.df[s] = self.df[s].astype(str)

    def print(self, max_rows=None, max_cols=None):
        with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols):
            print(self.df)

    def clear_data(self):
        self.df.dropna(how='any', inplace=True)
        self.df = self.df[(self.df['Gender'] == "male") | (self.df['Gender'] == "female")]
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print (df)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
class Base:
    def __init__(self, df):
        self.scl = pre.MinMaxScaler()
        self.df = df.copy()
        self.ind = self.df.columns.to_list()
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# jb, jbpv, skew, kurtosis = jarque_bera(self.model.wresid)
class DispAnalyzer(Base):
    def __init__(self, df,  dep, indep):
        Base.__init__(self, df)
        self.dep = dep
        self.indep = indep
        self.counter = 0
        self.tests = pd.DataFrame(
            columns=['statistic', 'p-value']
        )
        self.normality_tests = ['shapiro', 'kstest', 'jarque_bera', 'anderson']
        self.preprocessor = Preprocessor()

    def anderson(self, x):
        z = scipy.stats.anderson(x)
        n = len(x)
        ad_ = z[0] * (1 + .75/n + 2.25/n**2)
        p = None
        if ad_ >= 0.6:
            p = exp(1.2937 - 5.709*(ad_)+ 0.0186*(ad_**2))
        elif (ad_ < 0.6) and (ad_ > 0.34):
            p = exp(0.9177 - 4.279*(ad_) - 1.38*(ad_**2))
        elif (ad_ < 0.34) and (ad_ > 0.2):
            p = 1 - exp(-8.318 + 42.796*(ad_)- 59.938*(ad_**2))
        elif (ad_ < 0.2):
            p = 1 - exp(-13.436 + 101.14*(ad_)- 223.73*(ad_**2))
        return ad_, p

    def norm_tests(self):
        tests_f = {'shapiro': lambda x: scipy.stats.shapiro(list(x)),
                 'kstest': lambda x: scipy.stats.kstest(list(x), 'norm'),
                 'jarque_bera': lambda x: scipy.stats.jarque_bera(list(x)),
                   'anderson': lambda x: self.anderson(list(x))}
        for k in tests_f:
            self.tests.loc[k] = [*tests_f[k](self.df[self.dep])]

    def norm(self):
        return any(self.tests.loc[self.normality_tests,:] < 0.05)

    def histogram(self, color):
        # plt.figure(figsize=())
        fig, ax = plt.subplots(figsize = (6, 6))
        self.pngs += 1
        sns_plot = sns.distplot(self.df[self.dep], color=color, ax=ax)
        fig = sns_plot.get_figure()
        fig.savefig(f"app/static/distplot-{self.pngs}.jpg")
        plt.show()

    def normality(self):
        c = 0.5
        self.pngs = 0
        self.norm_tests()
        self.histogram('blue')
        if self.norm():
            self.df[self.dep] = self.df[[self.dep]].applymap(lambda x: np.log(x + 1))
            self.histogram('green')
            self.norm_tests()
        if self.norm():
            self.df[self.dep] = self.df[[self.dep]].applymap(lambda x: np.sqrt(x + c))
            self.histogram('yellow')
            self.norm_tests()
        if self.norm():
            self.df = self.df.iloc[0:100, :]
            self.histogram('red')
            self.norm_tests()

    def auto_correlation(self):
        self.dw = durbin_watson(self.residuals)  # ->2
        return self.dw

    def homoscedasticity(self, all=False):
        # sns.scatterplot(data=self.df, x=self.indep[0], y=self.dep)
        # plt.show()
        # lev, p_lev = scipy.stats.levene(*self.dfs)  # , p>0.05 good
        self.tests.loc['omnibus'] = [*omni_normtest(self.residuals)]
        self.tests.loc['normaltest'] = [*scipy.stats.normaltest(self.residuals)]

    def multicollinearities(self):
        return self.model.condition_number, 20  # should be lower than 20

    def set_levels(self, ind):
        uniques = self.df[self.indep[ind]].unique()
        self.dfs = []
        for j in uniques:
            self.dfs.append(self.df[self.df[self.indep[ind]] == j][self.dep])

    def anova_simple(self, ind=0):
        self.set_levels(ind)
        print(self.dfs)
        print(scipy.stats.f_oneway(*self.dfs))
        self.tests.loc['f_oneway'] = [*scipy.stats.f_oneway(*self.dfs)]

    def factor_analysis(self, op):
        right = f"{op}".join(map(lambda x: f"C({str(x)})", self.indep))
        self.op = op
        model_s = f"{self.dep} ~ {right}"
        self.model = ols(model_s, self.df).fit()
        self.residuals = self.model.wresid
        self.an_tb = []
        for k in [1, 2, 3]:
            self.an_tb.append(sm.stats.anova_lm(self.model, typ=k).drop(['Residual', 'Intercept'], errors='ignore'))

    def tukeyhsd(self, ind=0):
        mc = MultiComparison(self.df[self.dep], self.df[self.indep[ind]])
        mc_res = mc.tukeyhsd()  # alpha=0.1
        return mc_res

    def factor(self):
        self.factor_analysis('*')
        self.normality()
        self.homoscedasticity()
        self.auto_correlation()
        self.cond_num = self.multicollinearities()

    def texts(self):
        self.normality_text = self.preprocessor.text(self.tests, self.normality_tests, "о нормальности распределения")
        self.homoscedasticity_text = self.preprocessor.text(self.tests, ["omnibus"], "о наличии гомоскедостичности")
        self.auto_correlation_text = self.preprocessor.auto_correlation(self.dw)
        #self.tests, ["durbin-watson"], "о наличии авто-корреляции"
        self.multicollinearities_text = self.preprocessor.multicollinearities(self.cond_num)
        self.at = self.preprocessor.anova_tables(self.an_tb)

    def scenario_one_way(self):
        self.factor()
        self.texts()
        self.anova_simple()
        anova_test = self.preprocessor.text(self.tests, ["f_oneway"], f"о наличии зависимости {self.dep} от {self.indep}")
        return render_template("Analysis.html",
                               len=self.pngs,
                               normality=self.normality_text,
                               homoscedasticity=self.homoscedasticity_text,
                               auto_correlation=self.auto_correlation_text,
                               multicollinearities=self.multicollinearities_text,
                               anova_test=anova_test)

    def scenario_two_ways(self):
        self.factor()
        self.texts()
        anova_test1 = self.preprocessor.anova_multiply(self)
        self.factor_analysis('+')
        self.at2 = self.preprocessor.anova_tables(self.an_tb)
        anova_test2 = self.preprocessor.anova_multiply(self)
        tk1 = self.preprocessor.tukey(self.tukeyhsd(0))
        tk2 = self.preprocessor.tukey(self.tukeyhsd(1))
        print(self.tukeyhsd(0))
        print(self.tukeyhsd(1))
        return render_template("MultiplyAnalysis.html",
                               len=self.pngs,
                               normality=self.normality_text,
                               homoscedasticity=self.homoscedasticity_text,
                               auto_correlation=self.auto_correlation_text,
                               multicollinearities=self.multicollinearities_text,
                               anova_test = anova_test1,
                               anova_test2 = anova_test2,
                               anova_tables = self.at,
                               anova_tables2 = self.at2,
                               tukey = tk1,
                               tukey2 = tk2
                               )

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
class Preprocessor:
    def text(self, df, tests_list, hypothesis):
        s = []
        for k in tests_list:
                kk = df.loc[[k]]
                print(kk['statistic'].values)
                f1 = f"{k} тест: statistic = {kk['statistic'].values[0]:<1.4f}," \
                     f" p-value = {kk['p-value'].values[0]:<1.4f},"
                f2 = ""
                if kk['p-value'].values[0] < 0.05:
                    f2 += f"гипозета {hypothesis} отклоняется"
                else:
                    f2 += f"гипозета {hypothesis} не отклоняется"
                s.append((f1,f2))
        return s

    def multicollinearities(self, cond_number):
        f1 = f"Условное число: {cond_number[0]:<1.4f}, требуемое {cond_number[1]:<1.4f}; "
        f2 = ""
        if cond_number[0] < cond_number[1]:
            f2 += "Гипотеза о наличии мультиколлинеарности отклоняется"
        else:
            f2 += "гипозета о наличии мультиколлинеарности не отклоняется"
        return f1,f2

    def auto_correlation(self, st):
        f1 = f"durbin-watson тест: statistic = {st:<1.4f}, "
        f2 = ""
        if st < 2.5 and st > 1.5:
            f2 += "гипозета о наличии авто-корреляции отклоняется"
        else:
            f2 += "гипозета о наличии авто-корреляции не отклоняется"
        return f1, f2

    def anova_multiply(self, analyzer):
        f = ""
        t = 0
        for i in analyzer.an_tb:
            print(i)
            t += len(i[i['PR(>F)'] < 0.05])
        z = f"{analyzer.op}".join(analyzer.indep)
        if t > 0:
            f += f"Гипотеза о наличии зависимости {analyzer.dep} от {z} не отклоняется"
        else:
            f += f"Гипотеза о наличии зависимости {analyzer.dep} от {z} отклоняется"
        return f

    def anova_tables(self, an_tb):
        return [pht.build_table(an_tb[i], 'green_light').replace("\n", "") for i in range(3)]

    def tukey(self, tk):
        return tk.__str__().replace("\n","<br/>")

