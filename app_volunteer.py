# Import necessary libraries
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from gradientai import Gradient

# Load the dataset with a specified encoding
data = pd.read_csv('cleaneddata.csv', encoding='latin1')

# Page 1: Dashboard
def dashboard():
    st.image('logo.png', use_column_width=True)

    st.subheader("💡 Abstract: ")

    inspiration = '''
    The Edmonton Food Drive project intends to use machine learning to optimise drop-off and pick-up operations in order to revolutionise food donation management in Edmonton, Alberta. Current challenges in coordination and route planning necessitate automation for timely collections and reduced logistical complexities. Creating a machine learning model to determine the best drop-off sites, automating pick-up routes, and enhancing stakeholder communication are among the goals. The strategy include developing route planning algorithms, putting in place a centralised system for stakeholder cooperation, and evaluating historical data to determine the best sites. Lessons learnt include enhanced efficiency, improved resource utilization insights, and the value of real-time data in refining donation processes.
    '''

    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")

    what_it_does = '''
    We started a thorough data collecting phase utilising Google Forms for our food drive project, which was launched on September 23rd. Manual collection and Excel tools were also used for data collection. This project was made possible by the cooperative efforts of student organisations and Edmonton city volunteers. After that, in order to improve the quality and consistency of our dataset, we carefully cleaned the data, eliminating duplicates and standardizing formats to enhance the quality and reliability of our dataset. 
    Our machine learning goal is to solve a regression issue by predicting the number of donation bags that will be collected. This helps with the efficient resource utilization and allocation of volunteers. We created a user-friendly dashboard with tools for Exploratory Data Analysis (EDA), ML Modelling, Neighbourhood Mapping, Data Collection,  data visualizationas well as a chatbot for communicating with stakeholders. Among them the ML Modelling part helps  to predict number of donation bags in each stake, the EDA part offers insightful visualisations. A link to an updated Google Form is provided for volunteers in the Data Collection section, and the Neighbourhood Mapping option creates maps based on user inputs. The Chatbot enables easy communication among volunteers.
    '''

    st.write(what_it_does)
    st.subheader(" 🤝 Our partners:")
    
    st.image(["city.png","church.png","foodbank.png"],use_column_width = "never")



# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    
    grouped_data = data.groupby('Stake')['Donation Bags Collected'].sum().reset_index()

    # Plot the bar chart
    fig = px.bar(grouped_data, x='Stake', y='Donation Bags Collected', title='Total donation bags collected by stakes')
    st.plotly_chart(fig)

    # Visualize the distribution of numerical features using Plotly
    fig = px.histogram(data, x='# of Adult Volunteers', nbins=20, labels={'# of Adult Volunteers': 'Adult Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='# of Youth Volunteers', nbins=20, labels={'# of Youth Volunteers': 'Youth Volunteers'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='Donation Bags Collected', nbins=20, labels={'Donation Bags Collected': 'Donation Bags Collected'})
    st.plotly_chart(fig)

    fig = px.histogram(data, x='Time to Complete (min)', nbins=20, labels={'Time to Complete (min)': 'Time to Complete'})
    st.plotly_chart(fig)

# Page 3: Machine Learning Modeling
def machine_learning_modeling():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    completed_routes = st.slider("Completed More Than One Route", 0, 1, 0)
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
    youth_volunteers = st.slider("Number of Youth Volunteers", 0, 50, 10)

    # Predict button
    if st.button("Predict"):
        from sklearn.model_selection import train_test_split

        X = data.drop(columns=['Donation Bags Collected','Location','Ward/Branch','Stake','Unnamed: 0'])
        y = data['Donation Bags Collected']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
        model.fit(X_train, y_train)
    
        # Prepare input data for prediction
        user_input = [[adult_volunteers, youth_volunteers, time_spent, completed_routes,routes_completed , doors_in_route]] 

        # Make prediction
        prediction = np.round(model.predict(user_input))

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")

        # You can add additional information or actions based on the prediction if needed
# Page 4: Neighbourhood Mapping
# Read geospatial data
geodata = pd.read_csv("ADDRESS ONLY Property_Assessment_Data__Current_Calendar_Year_ - Property_Assessment_Data__Current_Calendar_Year_.csv")

def neighbourhood_mapping():
    st.title("Neighbourhood Mapping")

    # Get user input for neighborhood
    user_neighbourhood = st.text_input("Enter the neighborhood:")

    # Check if user provided input
    if user_neighbourhood:
        # Filter the dataset based on the user input
        filtered_data = geodata[geodata['Neighbourhood'] == user_neighbourhood]

        # Check if the filtered data is empty, if so, return a message indicating no data found
        if filtered_data.empty:
            st.write("No data found for the specified neighborhood.")
        else:
            # Create the map using the filtered data
            fig = px.scatter_mapbox(filtered_data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    hover_name='Neighbourhood',
                                    zoom=12)

            # Update map layout to use OpenStreetMap style
            fig.update_layout(mapbox_style='open-street-map')

            # Show the map
            st.plotly_chart(fig)
    else:
        st.write("Please enter a neighborhood to generate the map.")


def map_generation_google():
    st.title('Google Map')
    st.write('Here is an embedded edmonton ward Map:')

    # Embedding Google Map using HTML iframe
    st.markdown("""
     <iframe src="https://www.google.com/maps/d/u/0/embed?mid=1g_sjW6HnmJHMVdneYavOh5sLf-UKbyQ&ehbc=2E312F&noprof=1" width="640" height="480"></iframe>
    """, unsafe_allow_html=True)

# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    st.markdown("""
     <iframe src="https://docs.google.com/forms/d/e/1FAIpQLScurJAxkfSJSDfn3CXgGvJp8oPpv5kPCwACA1GR6vSLPTrONg/viewform?embedded=true" width="1024" height="768" frameborder="0" marginheight="0" marginwidth="0">Loading…</iframe>
    """, unsafe_allow_html=True)
    

# Page 6: Chatbot
os.environ['GRADIENT_ACCESS_TOKEN'] = st.secrets["GRADIENT_ACCESS_TOKEN"]
os.environ['GRADIENT_WORKSPACE_ID'] = st.secrets["GRADIENT_WORKSPACE_ID"]
def chatbot():
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")

        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # before fine-tuning
            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            st.markdown(f"Generated: {completion}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page", ["Dashboard", "Neighbourhood Mapping", "Google map generation", "Data Collection", "Chatbot"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling":
        machine_learning_modeling()
    elif app_page == "Neighbourhood Mapping":
        neighbourhood_mapping()
    elif app_page == "Google map generation":
        map_generation_google()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()
