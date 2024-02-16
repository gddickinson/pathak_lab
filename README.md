# Pathak lab PIEZO1 analysis tools
This repositoryy contains a selection of Python-based tools created to help the Pathak lab (https://www.pathaklab-uci.com/) analyse TIRF recordings of PIEZO1 using flika (https://flika-org.github.io/).

# Code organization 
The code is split into two sections: 'flika_plugins', containing flika extensions used to visualise and analyse TIRF recordings within flika; and 'flika_scripts', python scripts used to batch process multiple results files.

# Workflow
The analysis workflow depends on localization of PIEZO1 puncta using either the flika pynsight plugin (https://github.com/kyleellefsen/pynsight) or other localization software, such as thunderSTORM (https://zitmen.github.io/thunderstorm/), to identify fluorescent 'blobs' in every recording frame. Tracking is carried out using pynsight, and further analysis of point and track data performed on batches of loalization files using the scripts. Point and track data can be displayed using the plugins.
