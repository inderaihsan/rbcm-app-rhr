import streamlit as st
from datetime import date
from rcn import building_data, format_rupiah, estate_data, condition_format # Assuming you have a module named rcn.py with the RCN class

st.set_page_config(page_title="RCN Valuation App", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f4f4f4;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)


st.image("https://kjpp.rhr.co.id/wp-content/uploads/2020/12/LOGO_KJPP_RHR_1_resize.png", width=500 )
st.title("🏗️ RHR Building Cost Model")
st.markdown("Estimate the Indicative Market Value of a building using the cost approach.")

with st.form("valuation_form"):
    st.subheader("📋 Input Parameters")

    col1, col2 = st.columns(2)
    
    with col1:
        building_type = st.selectbox("Building Type", building_data.keys())
        kawasan = st.selectbox("Kawasan" , ["Estate", "Non Kawasan"])
        building_year = st.number_input("Building Year", min_value=1900, max_value=date.today().year, value=2010)
        valuation_date = st.date_input("Valuation Date", value=date.today())
        building_age = valuation_date.year - building_year
    
    with col2:
        gross_floor_area = st.number_input("Gross Floor Area (sqm)", min_value=0.0, value=10000.0)
        building_storey = st.number_input("Number of Storeys", min_value=1, value=2)
        extent_obsolescence = st.selectbox("Extent of Obsolescence", ["NO DAMAGES", "MINOR", "MAJOR", "UNHABITABLE"])
        condition = st.selectbox("Building Condition", [condition for condition in condition_format.keys()])

    submitted = st.form_submit_button("Calculate")

    if submitted:
        building_age = valuation_date.year - building_year 
        direct_cost = building_data[building_type]  # Placeholder for direct cost calculation 
        soft_cost = direct_cost * 0.145 
        price_before_tax = direct_cost + soft_cost
        price_after_tax = price_before_tax * 1.11
        profit_margin_coefficient = estate_data[building_type][kawasan] 
        profit_margin = price_after_tax * profit_margin_coefficient  
        rcn_per_sqm = price_after_tax + profit_margin
        rcn_area = rcn_per_sqm * gross_floor_area
        economic_building_age = estate_data[building_type]["building_age"] 
        if extent_obsolescence == "NO DAMAGES":
            obsolences = (economic_building_age - building_age)/economic_building_age 
        else : 
            obsolences = condition_format[condition] / 100

        st.success("Calculation submitted! Below are the results:")

        # Display results in a table format
        st.markdown("### 🏗️ Valuation Results")
        results = {
            "Building Age (years)": building_age,
            "Gross Floor Area (sqm)": gross_floor_area,
            "Direct Cost/sqm": format_rupiah(direct_cost),
            "Soft Cost/sqm": format_rupiah(soft_cost),
            "Price Before Tax": format_rupiah(price_before_tax),
            "Price After Tax": format_rupiah(price_after_tax),
            "Profit Margin": format_rupiah(profit_margin),
            "RCN/sqm": format_rupiah(rcn_per_sqm),
            "RCN": format_rupiah(rcn_area),
            "Condition Factor": f"{obsolences:.2%}"
        }

        # Create a table
        st.table(results)

        # Calculate the value of the property
        property_value = rcn_area * obsolences

        # Display the green bar
        st.markdown("### 💰 Property Value")
        st.success(f"The estimated property value is: {format_rupiah(property_value)}")
        
