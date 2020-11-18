# COVID-19 Mask Use (By US County)
### Author: Jace Kline

## Abstract
In this analysis, we aim to extract value from information concerning mask use for US counties. We compare this mask usage data to the cases and deaths trends over time for each county to determine the correlation between mask use and cases/deaths. In particular, we engineer a "mask score" for each county which we proceed to use in further correlations, plots, and analysis. We employ geographical plots of US counties to visually show mask scores and weekly average increases for each county. We conclude by clustering counties by mask score and population density.

Both of these datasets were accumulated by the New York Times and were pulled from Google Cloud Platform. In addition to these data sets, we incorporate data that associates each county with its area and population. This allows us to "standardize" the growth in cases/deaths among counties despite their difference in population, density, and area. This data set was pulled from GitHub at https://github.com/ykzeng/covid-19/blob/master/data/census-landarea-all.csv.

See the entire report [here](reports/mask-use.md).