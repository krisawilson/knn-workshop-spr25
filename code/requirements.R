# purpose: necessary packages for the workshop 

# one way: use the default install.packages() function.
# instead, going to use 'pak', since it's faster and more efficient:
install.packages("pak")

# now, here are the necessary packages to install:
packs <- c("tidyverse",
           "janitor",
           "caret",
           "FNN", 
           "tidytuesdayR")
pak::pkg_install(pkg = packs)

## in case pak doesn't like tidytuesdayR:
install.packages("tidytuesdayR")