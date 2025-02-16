#%%
import numpy as np
import rpy2
import rpy2.robjects as robjects
# from rpy2.rinterface import 

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects import r

numpy2ri.activate()
base = importr('base')
utils = importr('utils')
geoLift = importr('GeoLift')

# %%


print(geoLift)


# %%
geoLift.geoLift()

# %%
stats = importr('stats')
a=stats.rnorm(100)
print(a)

# %%
r.geoLift()
# %%
dat = r('data(GeoLift_PreTest)')


# %%



# %%

moo = r('''
GeoTestData_PreTest <- GeoDataRead(
  data = GeoLift_PreTest,
  date_id = "date",
  location_id = "location",
  Y_id = "Y",
  X = c(), # empty list as we have no covariates
  format = "yyyy-mm-dd",
  summary = TRUE
)
head(GeoTestData_PreTest)
''')
print(moo)

# %%
r['GeoLift_PreTest']

# %%
https://github.com/facebookincubator/GeoLift/blob/main/vignettes/GeoLift_Walkthrough.Rmd