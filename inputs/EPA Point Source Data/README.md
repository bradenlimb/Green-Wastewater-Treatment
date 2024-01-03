# Data Dictionary for Nutrient Loading Data for Wastewater Treatment Facilities in the United States

## Source
Original Source: [EPA Hypoxia Task Force Nutrient Model](https://echo.epa.gov/trends/loading-tool/hypoxia-task-force-nutrient-model)

## Files
- Nutrient-Modeling_Hypoxia_Task_Force_Search_clean.xlsx
- Nutrient-Modeling_Hypoxia_Task_Force_Search_clean_csv.csv
- Nutrient-Modeling_Hypoxia_Task_Force_Search_raw.csv

## Context
The data included in this data dictionary encompasses nutrient loading data of nitrogen and phosphorus at wastewater treatment facilities throughout the United States. Data was acquired through direct measurement and recording or was generated using the Hypoxia Task Force’s nutrient model.

The ‘Nutrient-Modeling_Hypoxia_Task_Force_Search_raw.csv’ file contains the raw data as provided by the EPA’s Nutrient Modeling from the Hypoxia Task Force. The ‘clean’ versions of the file remove the header and ensure the data is in the proper format to be imported by Python.

## Variable Descriptions
File: ‘Nutrient-Modeling_Hypoxia_Task_Force_Search_clean_csv.csv’

- **Year**: Year the data was recorded
- **NPDES Permit Number**: National Pollutant Discharge Elimination System (NPDES) permit number for each wastewater treatment facility
- **FRS ID**: Facility Registry System (FRS) identification number as assigned to the treatment facility by the EPA
- **SIC Code**: Standard Industrial Classification (SIC) code assigned to the wastewater treatment facility.
- **NAICS Code**: North American Industry Classification System (NAICS) code assigned to the wastewater treatment facility.
- **Facility Name**: Name of the wastewater treatment facility
- **City**: City name where the wastewater treatment facility is located.
- **State**: State name where the wastewater treatment facility is located.
- **ZIP Code**: ZIP code where the wastewater treatment facility is located.
- **County**: County where the wastewater treatment facility is located.
- **FIPS Code**: FIPS code for the county which the wastewater treatment facility is located.
- **EPA Region**: EPA region in which the wastewater treatment facility is located.
- **Facility Type**: Publicly owned or private treatment facility
    - POTW (publicly owned treatment works)
    - Non-POTW (non-publicly owned treatment works)
- **Permit Type**: Type of EPA permit that the wastewater treatment facility holds
- **Major/Non-Major Status**: Indicator representing major (≥1 MGD) and non-major ("minor" <1 MGD) municipal wastewater treatment facilities
- **HUC 12 Code**: Sub-watershed Hydrologic Unit Code as defined by the USGS where the wastewater treatment facility resides
- **Total Facility Design Flow (MGD)**: Design capacity of the wastewater treatment facility in millions of gallons per day
- **Actual Average Facility Flow (MGD)**: Mean daily flow for the wastewater treatment facility measured in millions of gallons per day.
- **Facility Latitude**: Latitude at which the wastewater facility is located.
- **Facility Longitude**: Longitude at which the wastewater facility is located.
- **Nutrient Type**: Nutrient type for which the data was recorded or modeled (Nitrogen or Phosphorus)
- **Total Pounds (lb/yr)**: Total pounds of nutrient released by the wastewater treatment facility each year.
- **Total Annual Flow (MGal/yr)**: Total annual flow of wastewater through the treatment facility each year measured in millions of gallons.
- **Max Allowable Load (lb/yr)**: The maximum allowable load of nutrients to leave the facility each year. Measured in pounds of nutrients per year. Not applicable to every facility.
- **Load Over Limit**: The number of pounds of nutrient released over the maximum allowable limit each year.
- **Avg Concentration**: The mean concentration of nutrients released in the wastewater from the treatment facility. Measured in milligrams of nutrient per liter of water.
- **Data Source for Concentration**: Type of data used for the concentration measurements. Direct measurement and reporting (DMR Data) or modeled data (Modeled).
- **Data Source for Flow**: Type of data used for the flow measurements. Direct measurement and reporting (DMR Data) or modeled data (Modeled).
- **Discharges to Impaired Water Bodies**: Identifies if the treatment facility discharges wastewater to an impaired water body as defined by the EPA’s 303d list. (Y/N)
- **Discharges Pollutants Contributing to Water Body Impairment**: Identifies if the treatment facility discharges nutrients to an impaired water body as defined by the EPA’s 303d list which contribute to further impairment of the
