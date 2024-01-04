**Data Dictionary for Results**

_Files:_
- results\_facility\_level-X\_HUC6.xlsx
- results\_ summaries\_level-X\_HUC6.xlsx
- summary\_statistics.xlsx


CONTEXT:
The data included in this data dictionary includes the results that were generated for the publication titled _Greening America's Rivers: The Potential of Climate Financing for Nature-Based Water Quality Improvement._ Similar results files will be generated when the python script is run.

The 'results\_facility\_level-X\_HUC6.xlsx' file contains raw results data from the nutrient trading analysis on the facility level for every green and gray treatment method for the specified treatment level X (levels 2, 3, 4, and 5 evaluated as specified by the [EPA](https://www.epa.gov/system/files/documents/2023-06/life-cycle-nutrient-removal.pdf)).

The 'results\_ summaries\_level-X\_HUC6.xlsx' file contains results data from the nutrient trading analysis aggregated on the national level for every green and gray treatment method for the specified treatment level X (levels 2, 3, 4, and 5 evaluated as specified by the [EPA](https://www.epa.gov/system/files/documents/2023-06/life-cycle-nutrient-removal.pdf)).

The 'summary\_statistics.xlsx' file contains summarized results as presented in the paper titled Greening America's Rivers: The Potential of Climate Financing for Nature-Based Water Quality Improvement.



VARIABLE DESCRIPTIONS:

File: 'results\_facility\_level-X\_HUC6.xlsx'

Sheet: 'keys'
- **Sheet Name** – Name of the sheet which includes the technology listed in the 'Technology' column
- **Technology** – Name of the technology used for nutrient treatment

Sheets: 1+
- **Facility** – National Pollutant Discharge Elimination System (NPDES) permit number for each wastewater treatment facility
- **HUC6** – Waterbasin Hydrologic Unit Code as defined by the USGS.
- **HUC8** – Sub-waterbasin Hydrologic Unit Code as defined by the USGS.
- **HUC10** – Watershed Hydrologic Unit Code as defined by the USGS.
- **HUC12** – Sub-watershed Hydrologic Unit Code as defined by the USGS.
- **eGrid** – Code designating which EPA eGRID electricity region that the facility belongs to.
- **N Total Load (kg/yr)** – Total nitrogen load released by the wastewater treatment facility each year measured in kilograms per year.
- **N Annual Flow (L/yr)** – Total annual wastewater flow through each facility measured in liters of water per year.
- **N Current Mean Conc (mg/L)** – Existing mean concentration of nitrogen in the wastewater measured in milligrams of nitrogen per liter.
- **N Load Treated (kg/yr)** - Total nitrogen load treated by the nutrient treatment technology each year measured in kilograms per year.
- **N Treatment Area (ha)** – Number of hectares required for nitrogen treatment using green treatment technologies.
- **N New Mean Conc (mg/L)** - Mean concentration of nitrogen in the wastewater measured in milligrams of nitrogen per liter after treatment.
- **N Conc Diff (mg/L)** – Difference in mean nitrogen concentration before and after treatment. Measured in milligrams of nitrogen per liter.
- **N Treated Cost ($/yr)** – Cost of nitrogen treatment annually. Measured in dollars per year.
- **N Treated GWP (tonnes-CO2eq/yr)** – Global warming potential of nitrogen treatment annually. Measured in tonnes of CO2 equivalent emissions per year.
- **P Total Load (kg/yr)** – Total phosphorus load released by the wastewater treatment facility each year measured in kilograms per year.
- **P Annual Flow (L/yr)** – Total annual wastewater flow through each facility measured in liters of water per year.
- **P Current Mean Conc (mg/L)** - Mean concentration of phosphorus in the wastewater measured in milligrams of phosphorus per liter.
- **P Load Treated (kg/yr)** - Total phosphorus load treated by the nutrient treatment technology each year measured in kilograms per year.
- **P Treatment Area (ha)** – Number of hectares required for phosphorus treatment using green treatment technologies.
- **P New Mean Conc (mg/L)** - Mean concentration of phosphorus in the wastewater measured in milligrams of phosphorus per liter after treatment.
- **P Conc Diff (mg/L)** – Difference in mean phosphorus concentration before and after treatment. Measured in milligrams of phosphorus per liter.
- **P Treated Cost ($/yr)** – Cost of phosphorus treatment annually. Measured in dollars per year.
- **P Treated GWP (tonnes-CO2eq/yr)** – Global warming potential of phosphorus treatment annually. Measured in tonnes of CO2 equivalent emissions per year.
- **Max Treated Cost ($/yr)** – Maximum cost of either nitrogen or phosphorus treatment annually, whichever is larger. Measured in dollars per year.
- **Max Treated GWP (tonnes-CO2eq/yr)** – Maximum global warming potential of either nitrogen or phosphorus treatment annually, whichever is larger. Measured in tonnes of CO2 equivalent emissions per year.

File: 'results\_ summaries\_level-X\_HUC6.xlsx'

- **Treatment** – Name of the technology used for nutrient treatment
- **Treatment Type –** Identifies if the technology was a Green or Gray treatment option
- **N Total Load (kg/yr)** – Total nitrogen load released each year measured in kilograms per year.
- **N Annual Flow (L/yr)** – Total annual wastewater flow measured in liters of water per year.
- **N Current Mean Conc (mg/L)** – Existing mean concentration of nitrogen in the wastewater measured in milligrams of nitrogen per liter.
- **N Load Treated (kg/yr)** - Total nitrogen load treated by the nutrient treatment technology each year measured in kilograms per year.
- **N Treatment Area (ha)** – Number of hectares required for nitrogen treatment using green treatment technologies.
- **N New Mean Conc (mg/L)** - Mean concentration of nitrogen in the wastewater measured in milligrams of nitrogen per liter after treatment.
- **N Conc Diff (mg/L)** – Difference in mean nitrogen concentration before and after treatment. Measured in milligrams of nitrogen per liter.
- **N Treated Cost ($/yr)** – Cost of nitrogen treatment annually. Measured in dollars per year.
- **N Treated GWP (tonnes-CO2eq/yr)** – Global warming potential of nitrogen treatment annually. Measured in tonnes of CO2 equivalent emissions per year.
- **P Total Load (kg/yr)** – Total phosphorus load released each year measured in kilograms per year.
- **P Annual Flow (L/yr)** – Total annual wastewater flow measured in liters of water per year.
- **P Current Mean Conc (mg/L)** - Mean concentration of phosphorus in the wastewater measured in milligrams of phosphorus per liter.
- **P Load Treated (kg/yr)** - Total phosphorus load treated by the nutrient treatment technology each year measured in kilograms per year.
- **P Treatment Area (ha)** – Number of hectares required for phosphorus treatment using green treatment technologies.
- **P New Mean Conc (mg/L)** - Mean concentration of phosphorus in the wastewater measured in milligrams of phosphorus per liter after treatment.
- **P Conc Diff (mg/L)** – Difference in mean phosphorus concentration before and after treatment. Measured in milligrams of phosphorus per liter.
- **P Treated Cost ($/yr)** – Cost of phosphorus treatment annually. Measured in dollars per year.
- **P Treated GWP (tonnes-CO2eq/yr)** – Global warming potential of phosphorus treatment annually. Measured in tonnes of CO2 equivalent emissions per year.
- **Max Treated Cost ($/yr)** – Maximum cost of either nitrogen or phosphorus treatment annually, whichever is larger. Measured in dollars per year.
- **Max Treated GWP (tonnes-CO2eq/yr)** – Maximum global warming potential of either nitrogen or phosphorus treatment annually, whichever is larger. Measured in tonnes of CO2 equivalent emissions per year.

File: 'summary\_statistics.xlsx'

Sheet: 'Statistics'
- **Cells A1-E15** – Results presented in Table 1 in the paper.
- **Cells G1-K15** – Statistics on the economic feasibility of using green treatment methods instead of gray treatment methods as a percentage of HUC6 waterbasin.

Sheets: Level X (Where 'X' represents a value between 2-5)
- **HUC6** – Waterbasin Hydrologic Unit Code as defined by the USGS.
- **Gray Cost ($/yr)** – Total cost of nutrient treatment in the waterbasin using gray treatment methods.
- **Gray Emissions (tonnes-CO2eq/yr)** – Total emissions of nutrient treatment in the waterbasin using gray treatment methods.
- **Green Cost ($/yr)** – Total cost of nutrient treatment in the waterbasin using green treatment methods.
- **Green Emissions (tonnes-CO2eq/yr)** – Total emissions of nutrient treatment in the waterbasin using green treatment methods.
- **Emissions Saved (tonnes-CO2eq/yr)** – Total emissions saved by using green treatment methods instead of gray treatment methods for nutrient treatment in the waterbasin.
- **N Total Load (kg/yr)** – Total nitrogen load released by the wastewater treatment facilities each year in each waterbasin measured in kilograms per year.
- **N Annual Flow (L/yr)** – Total annual wastewater flow through the wastewater treatment facilities in each waterbasin measured in liters of water per year for plants that treated nitrogen.
- **N Load Treated (kg/yr)** - Total nitrogen load treated by the nutrient treatment technologies each year in each waterbasin measured in kilograms per year.
- **P Total Load (kg/yr)** – Total phosphorus load released by the wastewater treatment facilities each year in each waterbasin measured in kilograms per year.
- **P Annual Flow (L/yr)** – Total annual wastewater flow through the wastewater treatment facilities in each waterbasin measured in liters of water per year for plants that treated phosphorus.
- **P Load Treated (kg/yr)** - Total phosphorus load treated by the nutrient treatment technologies each year in each waterbasin measured in kilograms per year.
- **N Load Treatment Needed (kg/yr)** – Total nitrogen load that needed to be treated in each waterbasin to meet desired concentration limits measured in kilograms per year.
- **N Treated Percent (kg/yr)** – Percent of N Load TreatmentNeeded which was treated in each waterbasin.
- **P Load Treatment Needed (kg/yr)** – Total phosphorus load that needed to be treated in each waterbasin to meet desired concentration limits measured in kilograms per year.
- **P Treated Percent (kg/yr)** – Percent of P Load TreatmentNeeded which was treated in each waterbasin.
- **Carbon Financing ($/yr)** – The total carbon financing in each waterbasin assuming a $20/tonne-CO2e carbon credit cost.
- **Percent Costs (%)** – The percent of the green treatment costs which could be paid for through carbon financing.
- **Column T** – The difference between the gray and green treatment costs in each waterbasin in dollars per year.
- **Column U** – The difference between the gray and green treatment costs including carbon financing in each waterbasin in dollars per year.
- **Cells W1-Y3** – Calculating the total electricity use for nutrient treatment for gray treatment methods.

Sheets: Level X Totals (Where 'X' represents a value between 2-5)
- **Cost ($/yr)** – Total costs for each treatment method across all waterbasins in the contiguous United States.
- **Emissions (tonnes-CO2eq/yr)** - Total emissions for each treatment method across all waterbasins in the contiguous United States.