#bigmart sales prediction

"""To run the app, you need to install streamlit [ (pip install streamlit) in your terminal ]
   Go to the folder where app.py (python script is saved).
   Open terminal at the folder and type:  streamlit run app.py

   You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
"""

html = """
  <style>
  u{
  text-decoration-line: underline;
  text-decoration-style: double;
  }

  mark{
  font-weight: bold; 
  background-color: yellow;
  color: black;
  }

  font{
   color: grey;
  }
  
}
    
  </style>
"""

#from st_collapsible_container_spec import * 
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
# import graphviz
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
def main():
	@st.cache(persist=True)
	def load_data():
		data=pd.read_csv("data.csv")
		return data



	def fig2img(fig):
	    import io
	    buf = io.BytesIO()
	    fig.savefig(buf)
	    buf.seek(0)
	    img = Image.open(buf)
	    return img

	@st.cache(persist=True)
	def split(df,test_size):
		y=df.Item_Outlet_Sales
		x=df.drop(columns=['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
		x_train,x_test,y_train,_y_test=train_test_split(x,y,test_size=test_size,random_state=0)
		return x_train,x_test,y_train,_y_test

	def show_split(x_train,x_test,test_size):
		st.subheader("Set the test size for spliting the data in to train and test set.")
		train_split=(1-test_size)*100
		test_split=test_size*100		
		if st.checkbox("show split",False):
			st.markdown("So the train-test split is: <b><u> %d-%d </u></b>" % (train_split,test_split), unsafe_allow_html=True)
			st.markdown("Train shape: <b>(%d,%d) </b>" % (x_train.shape),unsafe_allow_html=True)
			st.markdown("Test shape: <b> (%d,%d) </b>" % (x_test.shape),unsafe_allow_html=True)
	

	st.markdown(html, unsafe_allow_html=True)

	st.title("Regression Web App")
	st.sidebar.title("BigMart Sales Prediction💰")
	st.sidebar.subheader("Steps: ")
	
	df=load_data()

	if st.sidebar.checkbox("Show raw dataset",True):
		st.subheader("Bigmart Sales dataset: ")
		st.write(df)
		st.markdown("This is modified data")

	st.sidebar.subheader(" Choose Machine learning algo:  ")
	algo=st.sidebar.selectbox("Supervised ML Algos",("--select-- ","LinearRegression","DecisionTreeRegressor","RandomForestRegressor"))
	

	if algo=='LinearRegression':
		st.write("<h2><b><font>Linear Regression</b></h2>",unsafe_allow_html=True)
		test_size=st.slider("test size",0.1,0.5,0.01,key="test_size")

		x_train,x_test,y_train,y_test=split(df,test_size)
		show_split(x_train,x_test,test_size)

		lr=LinearRegression().fit(x_train,y_train)
		y_predictions=lr.predict(x_test)

		st.header("Linear Regression Results: ")

		c1,c2,c3=st.columns(3)

		with c1:
			r2_scoree=r2_score(y_predictions,y_test)
			st.write("<b> r2_score is: <br><mark> %f </mark></b>" % (r2_scoree),unsafe_allow_html=True)
		with c2:
			adjusted_r2=1- (((1-r2_scoree)*(len(x_test)-1))/(len(x_test)-len(x_test.columns)-1))
			st.write("<b> adjusted_r2 score is: <mark> %f </mark></b>" % (adjusted_r2),unsafe_allow_html=True)
		with c3:
			mean_squared_error_=mean_squared_error(y_predictions,y_test)
			st.write("<b> root mean_squared_error is: <mark> %f </mark></b>" % np.sqrt(mean_squared_error_),unsafe_allow_html=True)

		st.subheader("Feature Importance: ")
		e = st.expander("")
		coef1 = pd.Series(lr.coef_,x_train.columns).sort_values()
		
		fig,ax=plt.subplots()
		ax=coef1.plot(kind='bar', title='Model Coefficients')
		e.pyplot(fig=fig, clear_figure=None)

		st.subheader("Regression Plot")
		e = st.expander("")
		x_bins=e.number_input("x_bins",10,100,10)
		fig,ax=plt.subplots()
		ax=sns.regplot(y_test, lr.predict(x_test),robust=True,color="black",x_bins=x_bins)
		e.pyplot(fig=fig, clear_figure=None)

		st.subheader("Residual Plot")
		e = st.expander("")		
		fig,ax=plt.subplots()
		ax=sns.residplot(y_test, lr.predict(x_test),robust=True,color="red")
		e.pyplot(fig=fig, clear_figure=None)
	

	if algo=='DecisionTreeRegressor':
		st.write("<h2><b><font>Decision Tree Regressor</font></b></h2>",unsafe_allow_html=True)
		test_size=st.slider("test size",0.1,0.5,0.01,key="test_size")

		x_train,x_test,y_train,y_test=split(df,test_size)
		show_split(x_train,x_test,test_size)

		st.header("Model Hyperparameters")
		max_depth=st.number_input("The maximum depth of the tree",2,100,2)
		min_samples_leaf=st.number_input("Min samples leaf",20,400,20)

		dt=DecisionTreeRegressor().fit(x_train,y_train)
		y_predictions=dt.predict(x_test)

		st.header("DecisionTreeRegressor Results: ")

		c1,c2,c3=st.columns(3)

		with c1:
			r2_scoree=r2_score(y_predictions,y_test)
			st.write("<b> r2_score is: <br><mark> %f </mark></b>" % (r2_scoree),unsafe_allow_html=True)
		with c2:
			adjusted_r2=1- (((1-r2_scoree)*(len(x_test)-1))/(len(x_test)-len(x_test.columns)-1))
			st.write("<b> adjusted_r2 score is: <mark> %f </mark></b>" % (adjusted_r2),unsafe_allow_html=True)
		with c3:
			mean_squared_error_=mean_squared_error(y_predictions,y_test)
			st.write("<b> root mean_squared_error is: <mark> %f </mark></b>" % np.sqrt(mean_squared_error_),unsafe_allow_html=True)
		
		st.subheader("Feature Importance: ")
		e = st.expander("")
		coef1 = pd.Series(dt.feature_importances_,x_train.columns).sort_values()		
		fig,ax=plt.subplots()
		ax=coef1.plot(kind='bar', title='Model Coefficients')
		e.pyplot(fig=fig, clear_figure=None)

		st.subheader("Regression Plot")
		e = st.expander("")
		x_bins=e.number_input("x_bins",10,100,10)
		fig,ax=plt.subplots()
		ax=sns.regplot(y_test, dt.predict(x_test),robust=True,color="black",x_bins=x_bins)
		e.pyplot(fig=fig, clear_figure=None)

		st.subheader("Residual Plot")
		e = st.expander("")
		fig,ax=plt.subplots()
		ax=sns.residplot(y_test, dt.predict(x_test),robust=True,color="red")
		e.pyplot(fig=fig, clear_figure=None)

		st.header("Visualizing Decision Tree")
		e = st.expander("")
		fig,ax=plt.subplots(figsize=(9,13))
		fig.tight_layout()
		ax=tree.plot_tree(dt,max_depth=max_depth,filled=True,fontsize=14)
		img = fig2img(fig)
		st.image(img)

		st.write("<hr>",unsafe_allow_html=True)

	if algo=='RandomForestRegressor':
			st.write("<h2><font><b>Random Forest Regressor</b></font></h2>",unsafe_allow_html=True)
			test_size=st.slider("test size",0.1,0.5,0.01,key="test_size")

			x_train,x_test,y_train,y_test=split(df,test_size)
			show_split(x_train,x_test,test_size)

			st.header("Model Hyperparameters")
			n_estimators=st.number_input("Number of trees in the random forest",50,500,100)
			max_depth=st.number_input("The maximum depth of the tree",2,100,2)
			min_samples_leaf=st.number_input("Min samples leaf",20,400,20)

			rf=RandomForestRegressor(n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf).fit(x_train,y_train)
			y_predictions=rf.predict(x_test)

			st.header("RandomForestRegressor Results: ")

			c1,c2,c3=st.columns(3)

			with c1:
				r2_scoree=r2_score(y_predictions,y_test)
				st.write("<b> r2_score is: <br><mark> %f </mark></b>" % (r2_scoree),unsafe_allow_html=True)
			with c2:
				adjusted_r2=1- (((1-r2_scoree)*(len(x_test)-1))/(len(x_test)-len(x_test.columns)-1))
				st.write("<b> adjusted_r2 score is: <mark> %f </mark></b>" % (adjusted_r2),unsafe_allow_html=True)
			with c3:
				mean_squared_error_=mean_squared_error(y_predictions,y_test)
				st.write("<b> root mean_squared_error is: <mark> %f </mark></b>" % np.sqrt(mean_squared_error_),unsafe_allow_html=True)
			
			st.subheader("Feature Importance: ")
			e = st.expander("")
			coef1 = pd.Series(rf.feature_importances_,x_train.columns).sort_values()		
			fig,ax=plt.subplots()
			ax=coef1.plot(kind='bar', title='Model Coefficients')
			e.pyplot(fig=fig, clear_figure=None)

			st.subheader("Regression Plot")
			e = st.expander("")
			x_bins=e.number_input("x_bins",10,100,10)
			fig,ax=plt.subplots()
			ax=sns.regplot(y_test, rf.predict(x_test),robust=True,color="black",x_bins=x_bins)
			e.pyplot(fig=fig, clear_figure=None)

			st.subheader("Residual Plot")
			e = st.expander("")
			fig,ax=plt.subplots()
			ax=sns.residplot(y_test, rf.predict(x_test),robust=True,color="red")
			e.pyplot(fig=fig, clear_figure=None)

			st.header("Visualizing Random Forest")
			fig,ax=plt.subplots(figsize=(9,13))
			fig.tight_layout()
			tree_no=st.number_input("Select tree no. to plot: ",1,200,1)
			ax=tree.plot_tree(rf.estimators_[tree_no],max_depth=max_depth,filled=True,fontsize=14)
			img = fig2img(fig)
			st.image(img)

			st.write("<hr>",unsafe_allow_html=True)
    


	st.write("<hr>",unsafe_allow_html=True)




if __name__ == '__main__':
    main()

