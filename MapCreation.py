import folium.raster_layers
import netCDF4 as nc4
import cftime
import numpy as np
import xarray as xr
import datetime as dt
from datetime import timedelta
import matplotlib as mpl
from matplotlib import pyplot as plt

import matplotlib.dates as mdates
import united_states
import math
import pandas as pd
# import smogn
import rioxarray as rio
import copy
import seaborn as sns
import tables
import h5py
import rioxarray as rio
import rasterio
from rasterio.enums import Resampling
us = united_states.UnitedStates()
import reverse_geocoder as rg
import branca
import jinja2
from flask import Flask, render_template, request
import branca.colormap as cm
import folium
import altair as alt


Em = xr.open_dataset("Datasets\era5\CAdata_gridded(era5).nc")
EF = {"CO2":[1403,109, 1492,116, 1571,64],
        "CO":[121.9,54, 79,57.4, 38.1, 31.6],
        "Methane":[5.67,2.57, 4.31,3.04, 2.48,1.82],
        "Ethane":[1.022,0.411, 0.808,0.514, 0.458,0.352],
        "Acetylene":[0.0914,0.1075, 0.0979,0.0680, 0.0528,0.0277],
        "Benzene":[0.885,0.416, 0.571,0.437, 0.268,0.217],
        "Toulene":[1.412,0.89, 0.708,0.58, 0.292,0.244],
        "C8 Aromatics":[0.228,0.122, 0.140,0.103, 0.0764,0.064],
        "Isoprene and Pentadiene":[0.526,0.252, 0.307,0.208, 0.426,0.417],
        "Monterpenes":[4.016,3.744, 0.981,0.898, 0.566,0.349],
        "Phenol":[1.103,0.718, 0.776,0.717, 0.439,0.456],
        "Benzenediol and methyl furfural":[0.678,0.577, 0.430,0.421, 0.188,0.189],
        "Guaiacol":[0.651,0.505, 0.344,0.369, 0.154,0.160],
        "Syringol":[0.0121,0.0112, 0.00692,0.00493, 0.00427,0.00262],
        "Formaldehyde":[0.870,0.630, 0.45,0.426, 0.425,0.395],
        "Acetaldehyde":[0.246,0.14, 0.143,0.096, 0.098,0.086],
        "Acrolein":[0.341,0.308, 0.242,0.177, 0.142,0.114],
        "Acetone and Propanal":[1.20,0.56, 0.66,0.467, 0.438,0.337],
        "MVK and MACR":[0.272,0.212, 0.177,0.134, 0.106,0.081],
        "MEK and Butanals":[0.317,0.161, 0.199,0.148, 0.119,0.097],
        "Butanedione and Isomers":[0.328,0.296, 0.199,0.168, 0.110,0.086],
        "Furan":[0.648,0.356, 0.412,0.318, 0.194,0.163],
        "HCN":[0.695,0.414, 0.322,0.282, 0.139,0.141],
        "Acetonitrile":[0.371,0.209, 0.235,0.180, 0.158,0.151],
        "Acrylonitrile":[0.0302,0.0150, 0.0200,0.0137, 0.0111,0.0101],
        "BC":[0.837,1.096, 0.709,1.217, 0.02,0],
        "OA":[13.74,8.23, 16.53,14.70, 5.12,4.46]
}
multiplier = 1e3
cmapset = mpl.colormaps["magma"]
# cmapset = mpl.colormaps["viridis"]

app = Flask(__name__)
# print(GEm.Emissions[0][0][0].values != 0 and not np.isnan(GEm.Emissions[0][0][0]))
# print(np.nonzero(GEm.Emissions.values))
# truec = 0
# nonzeroes = np.nonzero(Em.DM[0].values)
def style(feature):
    norm = mpl.colors.Normalize(vmin=np.min(float(feature['geometry']['properties']['minval']),0), vmax=0.75*float(feature['geometry']['properties']['maxval']))
    color = mpl.colors.to_hex(cmapset(norm(float(feature['geometry']['properties']['value']))))
    return {'fillColor': color, 'color': "black", 'fillOpacity': 1.5*norm(float(feature['geometry']['properties']['value'])), 'weight':0.5}
    # return {'fillColor': color, 'color': "black", 'fillOpacity': 0.7, 'weight':0.5}

@app.route('/switchOverlay')
def switchOverlay():
    Months = ['January', 'February', 'March', 'April', 'May', "June", "July", "August", "September", "October", "November", "December"]
    Years = list(set(pd.to_datetime(Em['time'].values).year))
    Emissions = list(EF.keys())
    Scales = ['No Scale', 'Log Scale']
    min_lon, max_lon = -124.410607, -114.134458
    min_lat, max_lat = 32.534231, 42.009659
    m = folium.Map(
    max_bounds=True,    
    min_lat=min_lat,
    max_lat=max_lat,
    min_lon=min_lon,
    max_lon=max_lon,
    control_scale=True,
    zoom_control=False,
    scrollWheelZoom=False,
    dragging=False,
    double_click_zoom=False,
    zoom_start=7,
    # max_zoom=7,
    # min_zoom=10,
    location=[36.7783, -119.4179], 
    tiles=None,
    width="100vw",
    height="100vh",
    )
    folium.raster_layers.TileLayer(tiles='openstreetmap', control=False).add_to(m)

    yearfound = request.args.get('year')
    monthfound = request.args.get('month')
    emissionfound = request.args.get('emission')
    scalefound = request.args.get('scale')

    time = 12*Years.index(int(yearfound)) + Months.index(monthfound)
    Emission = emissionfound
    ef = np.mean([EF[Emission][0],EF[Emission][2]])
    values = Em.DM.values*ef
    if(scalefound == "Log Scale"):
        for row in range(0,len(values[time])):
            for col in range(0,len(values[time][row])):
                if(values[time][row][col]!= 0 and not np.isnan(values[time][row][col])):
                   values[time][row][col] = np.log(values[time][row][col]*multiplier)
    # folium.raster_layers.ImageOverlay(name = Emission +" Emissions", image=values, bounds = [[np.min(Em.DM.lat.values)-0.25, np.min(Em.DM.lon.values)-0.25], [np.max(Em.DM.lat.values)+0.25, np.max(Em.DM.lon.values)+0.25]], colormap= mpl.colormaps.get_cmap('viridis'), show=True).add_to(m)
    # folium.FitOverlays().add_to(m)
    for lat in range(0, len(Em.DM["lat"])):
        for lon in range(0, len(Em.DM["lon"])):
            if(not np.isnan(values[time][lat][lon])):
                # gj = folium.GeoJson(data= {"type": "Polygon", "coordinates": [[[Em.DM["lon"][lon].values-0.25, Em.DM["lat"][lat].values-0.25], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.25], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values-0.25, Em.DM["lat"][lat].values-0.125]]]})
                gj = folium.GeoJson(data= {"type": "Polygon", "coordinates": [[[Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values+0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values+0.125, Em.DM["lat"][lat].values+0.125], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values+0.125]]],"properties":{'value': str(values[time][lat][lon]) , 'maxval':str(np.nanmax(values[time])), 'minval':str(np.nanmin(values[time]))}}, style_function = style)
                # gj.add_child(folium.Popup(str(values[lat][lon]), sticky=True))
                # iframe = branca.element.IFrame(html = render_template('iframe.html', lat= lat,lon=lon, value = values[lat][lon]), width =100, height = 100)
                variablevals = pd.DataFrame({'varnames': Em['variables'].values, 'values': Em.data[time][lat][lon].values})
                chart1 = alt.Chart(variablevals).mark_bar().encode(x='varnames', y = 'values', tooltip='values')
                # print(Em['time'].values[max(time-5,0):min(time+6, len(Em.time.values))])
                # print(values[max(time-5,0):min(time+6, len(Em.time.values)), lat, lon])
                timeseries = pd.DataFrame({"time": Em['time'].values[max(time-6,0):min(time+6, len(Em.time.values))], 'emissions': values[max(time-6,0):min(time+6, len(Em.time.values)), lat, lon], 'month' : Em['time.month'].values[max(time-6,0):min(time+6, len(Em.time.values))]})
                overall = pd.DataFrame({'month': Em['time.month'].values, 'avg_emissions': values[:, lat,lon]})
                overall = overall.groupby('month').mean(numeric_only=True)
                timeseries = pd.merge(timeseries, overall, on='month', how='inner')
                avg = timeseries[['time', 'avg_emissions']].assign(type=['Overall Average Emission']*len(timeseries))
                avg = avg.rename(columns={'avg_emissions': 'emissions'})
                trueplot = pd.concat([timeseries[['time', 'emissions']].assign(type = ['Actual Emission']*len(timeseries)), avg])
                chart2 = alt.Chart(trueplot).encode(x = 'time', y = 'emissions', tooltip='emissions', color='type:N')
                chart2 = chart2.mark_line()+chart2.mark_point()
                iframe = branca.element.IFrame(html = render_template('iframe.html',   vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),), width =1100, height = 500)
                gj.add_child(folium.Popup(iframe, lazy= True))
                gj.add_to(m)
    # return m.get_root().render()
    Months.remove(monthfound)
    Years.remove(int(yearfound))
    Emissions.remove(emissionfound)
    Scales.remove(scalefound)
    colormap = cm.LinearColormap(colors = cmapset.colors, vmin=np.min(np.nanmin(values[time]),0), vmax=np.nanmax(values[time]), caption='Emission Quanitity')
    m.add_child(colormap)
    m.get_root().render()
    header = m.get_root().header.render()
    body_html = m.get_root().html.render()
    script = m.get_root().script.render()
    return render_template("Map.html", months=Months, emissions= Emissions, years = Years, selectedemission = emissionfound, selectedyear = yearfound, selectedmonth = monthfound, body_html=body_html, script=script,header=header, selectedscale = scalefound, scales = Scales)


@app.route('/')
def start():
    Months = ['January', 'February', 'March', 'April', 'May', "June", "July", "August", "September", "October", "November", "December"]
    Years = list(set(pd.to_datetime(Em['time'].values).year))
    Emissions = list(EF.keys())
    Scales = ['No Scale', 'Log Scale']
    min_lon, max_lon = -124.410607, -114.134458
    min_lat, max_lat = 32.534231, 42.009659
    m = folium.Map(
    max_bounds=True,    
    min_lat=min_lat,
    max_lat=max_lat,
    min_lon=min_lon,
    max_lon=max_lon,
    control_scale=True,
    zoom_control=False,
    scrollWheelZoom=False,
    dragging=False,
    double_click_zoom=False,
    zoom_start=7,
    # max_zoom=7,
    # min_zoom=10,
    location=[36, -119], 
    tiles=None,
    width="100vw",
    height="100vh",
    )

    folium.raster_layers.TileLayer(tiles='openstreetmap', control=False).add_to(m)
    monthfound = Months.pop(0)
    yearfound = Years.pop(0)
    emissionfound = Emissions.pop(0)
    scalefound = Scales.pop(0)

    time = 0
    Emission = emissionfound
    ef = np.mean([EF[Emission][0],EF[Emission][2]])
    values = Em.DM.values*ef
    # folium.raster_layers.ImageOverlay(name = Emission +" Emissions", image=values, bounds = [[np.min(Em.DM["lat"].values)-0.125, np.min(Em.DM["lon"].values)-0.125], [np.max(Em.DM["lat"].values)+0.125, np.max(Em.DM["lon"].values)+0.125]], colormap= mpl.colormaps.get_cmap('viridis'), show=True).add_to(m)
    # folium.FitOverlays().add_to(m)
    root = request.url_root
    for lat in range(0, len(Em.DM["lat"])):
        for lon in range(0, len(Em.DM["lon"])):
            if(not np.isnan(values[time][lat][lon])):
                # gj = folium.GeoJson(data= {"type": "Polygon", "coordinates": [[[Em.DM["lon"][lon].values-0.25, Em.DM["lat"][lat].values-0.25], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.25], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values-0.25, Em.DM["lat"][lat].values-0.125]]]})
                gj = folium.GeoJson(data= {"type": "Polygon", "coordinates": [[[Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values+0.125, Em.DM["lat"][lat].values-0.125], [Em.DM["lon"][lon].values+0.125, Em.DM["lat"][lat].values+0.125], [Em.DM["lon"][lon].values-0.125, Em.DM["lat"][lat].values+0.125]]],"properties":{'value': str(values[time][lat][lon]) , 'maxval':str(np.nanmax(values[time])), 'minval':str(np.nanmin(values[time]))}}, style_function = style)
                # gj.add_child(folium.Popup(str(values[lat][lon]), sticky=True))
                # iframe = branca.element.IFrame(html = render_template('iframe.html', lat= lat,lon=lon, value = values[lat][lon]), width =100, height = 100)
                variablevals = pd.DataFrame({'varnames': Em['variables'].values, 'values': Em.data[time][lat][lon].values})
                chart1 = alt.Chart(variablevals).mark_bar().encode(x='varnames', y = 'values', tooltip='values')
                # print(Em['time'].values[max(time-5,0):min(time+6, len(Em.time.values))])
                # print(values[max(time-5,0):min(time+6, len(Em.time.values)), lat, lon])
                timeseries = pd.DataFrame({"time": Em['time'].values[max(time-6,0):min(time+6, len(Em.time.values))], 'emissions': values[max(time-6,0):min(time+6, len(Em.time.values)), lat, lon], 'month' : Em['time.month'].values[max(time-6,0):min(time+6, len(Em.time.values))]})
                overall = pd.DataFrame({'month': Em['time.month'].values, 'avg_emissions': values[:, lat,lon]})
                overall = overall.groupby('month').mean(numeric_only=True)
                timeseries = pd.merge(timeseries, overall, on='month', how='inner')
                avg = timeseries[['time', 'avg_emissions']].assign(type=['Overall Average Emission']*len(timeseries))
                avg = avg.rename(columns={'avg_emissions': 'emissions'})
                trueplot = pd.concat([timeseries[['time', 'emissions']].assign(type = ['Actual Emission']*len(timeseries)), avg])
                chart2 = alt.Chart(trueplot).encode(x = 'time', y = 'emissions', tooltip='emissions', color='type:N')
                chart2 = chart2.mark_line()+chart2.mark_point()
                iframe = branca.element.IFrame(html = render_template('iframe.html',   vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                spec1=chart1.to_json(indent=None),
                spec2=chart2.to_json(indent=None),), width =1100, height = 500)
                gj.add_child(folium.Popup(iframe, lazy= True))
                gj.add_to(m)
    # return m.get_root().render()
    colormap = cm.LinearColormap(colors = cmapset.colors, vmin=np.min(np.nanmin(values[time]),0), vmax=np.nanmax(values[time]), caption='Emission Quanitity')
    m.add_child(colormap)
    m.get_root().render()
    header = m.get_root().header.render()
    body_html = m.get_root().html.render()
    script = m.get_root().script.render()
    return render_template("Map.html", months=Months, emissions= Emissions, years = Years, selectedemission = emissionfound, selectedyear = yearfound, selectedmonth = monthfound, body_html=body_html, script=script,header=header, selectedscale = scalefound, scales = Scales)
    # response =  render_template("Map.html", months=Months, emissions= Emissions, years = Years, selectedemission = emissionfound, selectedyear = yearfound, selectedmonth = monthfound) + m.get_root().render()
    # return response

if __name__ == "__main__":
    app.run(debug=True)