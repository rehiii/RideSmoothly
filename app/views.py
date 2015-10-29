from flask import render_template, request
from app import app
from routing import get_route
# import MySQLdb
# from a_Model import ModelIt


# @app.route('/')
# @app.route('/index')
# def index():
#     return "test!"

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/output')
def output():
    start = request.args.get('start')
    dest = request.args.get('dest')
    start_dest = '{0}_{1}'.format (start, dest)

    start_loc, dest_loc, distancekm, distancekm_shortest, linepaths, short_not_smooth, linepath_phdist, ph_info, ph_info_all, centerlat, centerlon, ph_info_dt = get_route (start, dest, ddist=0.02, min_dph=0.05)


    if dest != None:

        distancekm = '{0:.2f}'.format (distancekm[0])
        distancekm_shortest = distancekm_shortest[0]

        distancemi = '{0:.2f}'.format (float(distancekm) / 1.609)
        distancemi_shortest = distancekm_shortest / 1.609

        distance_ratio_float = float(distancekm) * 1.0 / float(distancekm_shortest)
        distance_ratio = '{0:.2f}'.format (distance_ratio_float)

        distance_diff_float = float(distancemi) - float(distancemi_shortest)
        distance_diff = '{0:.2f}'.format (distance_diff_float)

        linepath = linepaths[0]
        short_not_smooth = short_not_smooth[0]
        linepath_phdist = linepath_phdist[0]
        

        #print 'short_not_smooth {0}'.format (short_not_smooth)

        return render_template("output.html", distancekm = distancekm,
                               distancemi = distancemi,
                               distancekm_shortest = distancekm_shortest,
                               distancemi_shortest = distancemi_shortest,
                               distance_ratio = distance_ratio,
                               distance_diff = distance_diff,
                               start_add = start, 
                               start_lat = start_loc.latitude, 
                               start_lon = start_loc.longitude,
                               dest_add = dest,
                               dest_lat = dest_loc.latitude, 
                               dest_lon = dest_loc.longitude,
                               linepath = linepath,
                               short_not_smooth = short_not_smooth,
                               linepath_phdist = linepath_phdist,
                               ph_info = ph_info,
                               ph_info_all = ph_info_all,
                               centerlat = centerlat,
                               centerlon = centerlon,
                               ph_info_dt = ph_info_dt)
                               # ph_latlons = ph_latlons,
                               # open_ph_addresses = open_ph_addresses,
                               # open_ph_dates = open_ph_dates)
    else:

        return render_template("output.html", distancekm = None,
                               distancemi = None,
                               distancekm_shortest = None,
                               distancemi_shortest = None,
                               distance_ratio = None,
                               distance_diff = None,
                               start_add = start, 
                               start_lat = None, 
                               start_lon = None,
                               dest_add = dest,
                               dest_lat = None, 
                               dest_lon = None,
                               linepath = None,
                               short_not_smooth = None,
                               linepath_phdist = None,
                               ph_info = ph_info,
                               ph_info_all = ph_info_all,
                               centerlat = centerlat,
                               centerlon = centerlon,
                               ph_info_dt = ph_info_dt)
                               # ph_latlons = ph_latlons,
                               # open_ph_addresses = open_ph_addresses,
                               # open_ph_dates = open_ph_dates)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500
