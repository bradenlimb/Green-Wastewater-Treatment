# Nutrient Remediation Analysis: Green vs. Gray Treatment Methods

## Abstract
This Python script evaluates the comparative costs and emissions of both gray and green treatment methods for nutrient remediation in the contiguous United States. Our research focuses on the shift from conventional electricity-consuming gray infrastructure to nature-based solutions for improving in-stream water quality. The analysis encompasses a wide range of data, including impaired waters, treatment technologies, and life cycle greenhouse gas emission accounting across the contiguous United States.

We delve into traditional gray treatment technologies and juxtapose them with green technologies such as constructed wetlands, saturated buffers, and agricultural management practices. Our findings reveal that green alternatives are not only more cost-effective but also less energy and carbon-intensive compared to gray infrastructure. Implementing these green solutions across the contiguous United States could result in significant savings of $15.6 billion USD, reduce electricity usage by 21.2 terawatt-hours, and cut down 29.8 million tonnes of CO2-equivalent emissions per year. Additionally, these methods are capable of sequestering over 4.2 million CO2e annually over a 40-year period.

One of the key challenges in the adoption of green infrastructure is the inherent risk aversion of utilities and regulators. However, our analysis suggests that green solutions could generate approximately $679 million annually in carbon credit revenue (valued at $20/credit). This presents a unique opportunity to institutionalize green solutions that adhere to the same water quality standards as gray infrastructure.

## Script Functionality
- **Data Integration**: The script integrates data from various sources, focusing on nutrient loads, treatment technologies, and regional environmental data.
- **Comparative Analysis**: It conducts a comparative analysis between green and gray treatment methods in terms of cost, energy usage, and carbon emissions.
- **Financial & Environmental Impact**: Calculates potential savings, electricity reduction, and emission mitigation through the adoption of green technologies.

## How to Use
1. **Installation**: Clone the repository and install the required Python packages listed at the top of the `green_wastewater_treatment_point-source_data-reduction.py` file. This Python script was written in the Spyder IDE using an Anaconda environment. Depending on your Python installation, you might need to comment out lines 10-11. Those lines reset all of the variables before running the analysis and are unnecessary to complete the anlaysis.
2. **Data Input**: All inputs needed to run the code are found in the `inputs` folder. Input assumptions for the Green or Gray treatment methods can be updated in the `model_inputs.xlsx` spreadsheet.
3. **Execution**: Run the script using `python green_wastewater_treatment_point-source_data-reduction.py`.
4. **Results**: The script will output comparative results in terms of costs and emissions for each of the technologies which can be found in the `results` folder.
5. **Post Processing**: After the results are generated, the `csv_creation_for_images.py` file can be used to generate csv files that summarize the results and can be used for figure generation.

## Contributions and Feedback
We welcome contributions and feedback to improve this analysis. Please feel free to fork the repository, make changes, and submit a pull request. For feedback and suggestions, please open an issue in the repository.
