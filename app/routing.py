def get_route (start_address, dest_address, ddist=0.02, min_dph=0.05):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd

    import sys
    sys.path.append ('/home/ubuntu/python_adds')
    import cache
    import misc

    from datetime import datetime as dt

    ## Get pothole ids for graph node input
    @np.vectorize
    def get_ph_ids (service_num):
        return 'ph{0}'.format (service_num[3:])

    # Vectorized Get datetime format function
    @np.vectorize
    def get_dt_obj (t_str, date_format):
        #print (t_report_str)
        dt_obj = dt.strptime (t_str, date_format)
        return dt_obj

    # Vectorized get datetime.timedelta object hours
    @np.vectorize
    def get_dt_hours (dt):
        dt_hours = (dt.days * 24) + (dt.seconds * 1.0 / (60*60))
        return dt_hours

    @np.vectorize
    def get_dt_days (dt):
        dt_days = (dt.days) + (dt.seconds * 1.0 / (60*60*24))
        return dt_days


    # Retrieve coordinates
    if dest_address == None:

        # Get pothole data
        all_phs_df = misc.loading ('/var/www/RideSmoothly/app/static/baltimore_reported_phs_08_01_2014_08_05_2015_loc_openstreet.df')
        status = all_phs_df['status'].str.lower ()
        open_phs_withnans = all_phs_df[-status.str.contains ('clos')]
        open_phs = open_phs_withnans[open_phs_withnans.failed == 0]
        open_ph_addresses = open_phs['address_complete'].values
        open_ph_dates = open_phs['createdDate'].values
        ph_ids = get_ph_ids (open_phs['servicerequestnum'].values)
        ph_lats = open_phs['latitude'].values
        ph_lons = open_phs['longitude'].values

        all_phs = all_phs_df[all_phs_df.failed == 0]
        ph_lats_all = all_phs['latitude'].values
        ph_lons_all = all_phs['longitude'].values

        ph_info = []
        for lat, lon, address, date in np.array ([ph_lats, ph_lons, open_ph_addresses, open_ph_dates]).T:
            ph_info.append ((lat, lon, address, date[:22]))

        ph_info_all = []
        for lat, lon in np.array ([ph_lats_all, ph_lons_all]).T:
            ph_info_all.append ([lat, lon])

        closed_phs_withnans = all_phs_df[status.str.contains ('clos')]
        closed_phs = closed_phs_withnans[closed_phs_withnans.failed == 0]
        closed_ph_addresses = closed_phs['address_complete'].values
        closed_ph_dates = closed_phs['createdDate'].values
        closed_ph_statdate = closed_phs['statusDate'].values
        closed_ph_lats = closed_phs['latitude'].values
        closed_ph_lons = closed_phs['longitude'].values

        date_format = '%m/%d/%Y %I:%M:%S %p +0000'
        t_reported = get_dt_obj (closed_ph_dates, date_format)
        t_closed = get_dt_obj (closed_ph_statdate, date_format)
        dt = t_closed - t_reported
        dt_hours = get_dt_hours (dt)
        dt_days = get_dt_days (dt)
        ph_info_dt = []
        # for lat, lon, hours in np.array ([closed_ph_lats, closed_ph_lons, dt_hours]).T:
        #     ph_info_dt.append ([lat, lon, hours])
        for lat, lon, days in np.array ([closed_ph_lats, closed_ph_lons, dt_days]).T:
            ph_info_dt.append ([lat, lon, days])


        return 1, 1, 1, 1, 1, 1, 1, ph_info, ph_info_all, 39.319826, -76.609502, ph_info_dt

    else:

        import geopy
        geocoder = geopy.geocoders.Nominatim()
        try:
            start_loc = geocoder.geocode (start_address)
        except:
            geocoder = geopy.geocoders.GoogleV3()
            start_loc = geocoder.geocode (start_address)
            dest_loc = geocoder.geocode (dest_address)
        else:
            dest_loc = geocoder.geocode (dest_address)

        #return start_loc, dest_loc

        pos0 = (start_loc.latitude, start_loc.longitude)
        pos1 = (dest_loc.latitude, dest_loc.longitude)

        import networkx as nx
        import get_osm_nx
        from itertools import islice

        import pymysql

        ## Get lat/lon mins/maxs for smopy map setter
        def get_mins (lats, lons):
            lat_min = np.min (lats)
            lon_min = np.min (lons)
            lat_max = np.max (lats)
            lon_max = np.max (lons)
            return ((lat_min, lon_min), (lat_max, lon_max))

        def get_node_coords(graph, n):
            """If n0 and n1 are connected nodes in the graph, this function
            returns an array of point coordinates along the road linking
            these two nodes."""
            lat = graph.node[n]['data'].lat
            lon = graph.node[n]['data'].lon
            return (lat, lon)

        def get_path_coords(graph, n0, n1):
            """If n0 and n1 are connected nodes in the graph, this function
            returns an array of point coordinates along the road linking
            these two nodes."""
            # return np.array (lat0, lon0, lat1, lon1)
            # way = sg[n0][n1]['data']
            # lats = [sg.node[n]['data'].lat for n in way.nds]
            # lons = [sg.node[n]['data'].lon for n in way.nds]
            # print (n0, n1)
            lat0 = graph.node[n0]['data'].lat
            lon0 = graph.node[n0]['data'].lon
            lat1 = graph.node[n1]['data'].lat
            lon1 = graph.node[n1]['data'].lon
            return ((lat0, lon0), (lat1, lon1))


        @np.vectorize
        def get_path_coords_vect(graph, n0, n1):
            """If n0 and n1 are connected nodes in the graph, this function
            returns an array of point coordinates along the road linking
            these two nodes."""
            # return np.array (lat0, lon0, lat1, lon1)
            # way = sg[n0][n1]['data']
            # lats = [sg.node[n]['data'].lat for n in way.nds]
            # lons = [sg.node[n]['data'].lon for n in way.nds]
            # print (n0, n1)
            lat0 = graph.node[n0]['data'].lat
            lon0 = graph.node[n0]['data'].lon
            lat1 = graph.node[n1]['data'].lat
            lon1 = graph.node[n1]['data'].lon
            return lat0, lon0, lat1, lon1


        # Calc distance between any two points in geograph. coords (assume spherical Earth)
        # http://stackoverflow.com/questions/8858838/need-help-calculating-geographical-distance
        EARTH_R = 6372.8
        def geocalc(lat0, lon0, lat1, lon1):
            """Return the distance (in km) between two points in 
            geographical coordinates."""
            lat0 = np.radians(lat0)
            lon0 = np.radians(lon0)
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            dlon = lon0 - lon1
            y = np.sqrt(
                (np.cos(lat1) * np.sin(dlon)) ** 2
                 + (np.cos(lat0) * np.sin(lat1) 
                 - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
            x = np.sin(lat0) * np.sin(lat1) + \
                np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
            c = np.arctan2(y, x)
            return EARTH_R * c

        EARTH_R = 6372.8
        @np.vectorize
        def geocalc_vect(lat0, lon0, lat1, lon1):
            """Return the distance (in km) between two points in 
            geographical coordinates."""
            lat0 = np.radians(lat0)
            lon0 = np.radians(lon0)
            lat1 = np.radians(lat1)
            lon1 = np.radians(lon1)
            dlon = lon0 - lon1
            y = np.sqrt(
                (np.cos(lat1) * np.sin(dlon)) ** 2
                 + (np.cos(lat0) * np.sin(lat1) 
                 - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
            x = np.sin(lat0) * np.sin(lat1) + \
                np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
            c = np.arctan2(y, x)
            return EARTH_R * c

        def curry_geocalc_min (ph_lats, ph_lons):
            def geocalc_min_curried (lat0, lon0):
                return np.min(geocalc_vect (lat0, lon0, ph_lats, ph_lons))
            return geocalc_min_curried

        # Define a function that concats the positions of the points along
        # every edge in the right order along the path.  The order is based on
        # the fact that the last point in an edge needs to be close to the
        # first point in the next edge.
        def get_full_path(graph, path):
            """Return the positions along a path."""
            p_list = []
            curp = None
            for i in range(len(path)-1):
                p = np.array (get_path_coords (graph, path[i], path[i+1]))
                if curp is None:
                    curp = p
                if np.sum((p[0]-curp)**2) > np.sum((p[-1]-curp)**2):
                    p = p[::-1,:]
                p_list.append(p)
                curp = p[-1]
            return np.vstack(p_list)

        # No order specification
        # def get_full_path(path):
        #     """Return the positions along a path."""
        #     p_list = []
        #     for i in range(len(path)-1):
        #         p = np.array (get_path_coords (path[i], path[i+1]))
        #         p_list.append(p)
        #     return np.vstack(p_list)

        # Produce generator for k shortest paths
        def k_shortest_paths(G, source, target, k, weight=None):
            return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

        def pos_lims (pos0, pos1):
            minlat = min(pos0[0], pos1[0])
            maxlat = max(pos0[0], pos1[0])
            minlon = min(pos0[1], pos1[1])
            maxlon = max(pos0[1], pos1[1])
            center = (np.mean([pos0[0], pos1[0]]), np.mean([pos0[1], pos1[1]]))
            return minlat, maxlat, minlon, maxlon, center[0], center[1]

        # Make coordinates strings for plot names
        def coord_str (pos):
            str_lat = str(pos[0])
            new_str_lat = str_lat.replace ('.', 'd')
            str_lon = str(pos[1])
            new_str_lon = str_lon.replace ('.', 'd')
            return '{0}_{1}'.format (new_str_lat, new_str_lon)

        pos0_str = coord_str (pos0)
        pos1_str = coord_str (pos1)

        # Load graph and edge data

        minlat, maxlat, minlon, maxlon, centerlat, centerlon = pos_lims (pos0, pos1)

        dbcon = pymysql.connect ('localhost', 'root', 'Wt2Vr3!', 'ridesmoothly')
        cur = dbcon.cursor ()
        # Nodes
        nodes_dict = {}
        node_colnames = ('ID', 'LAT', 'LON')
        n_nodes = -1
        for colname in node_colnames:
            cur.execute (""" SELECT {0} FROM nodes WHERE LAT > {1} AND LAT < {2} AND LON > {3} AND LON < {4}""".format (colname, minlat-.01, maxlat+.01, minlon-.01, maxlon+.01))
            data = cur.fetchall ()
            col = [item[0] for item in data]
            nodes_dict[colname] = col
            if n_nodes < 0:
                n_nodes = len (col)
            else:
                assert (len (col) == n_nodes)
        # Edges
        edges_dict = {}
        edge_colnames = ('IDINT', 'NODE0', 'NODE1', 'LAT0', 'LON0', 'LAT1', 'LON1', 'LENGTH', 'MINPHDISTOPEN')
        n_edges = -1
        for colname in edge_colnames:
            cur.execute (""" SELECT {0} FROM edges WHERE LAT0 > {1} AND LAT1 > {1} AND LAT0 < {2} AND LAT1 < {2} AND LON0 > {3} AND LON1 > {3} AND LON0 < {4} AND LON1 < {4}""".format (colname, minlat-.01, maxlat+.01, minlon-.01, maxlon+.01))
            data = cur.fetchall ()
            col = [item[0] for item in data]
            edges_dict[colname] = col
            if n_edges < 0:
                n_edges = len (col)
            else:
                assert (len (col) == n_edges)


        g = nx.Graph()
        for node_id, node_lat, node_lon in zip (nodes_dict['ID'], nodes_dict['LAT'], nodes_dict['LON']):
            g.add_nodes_from ([(node_id, dict (data=get_osm_nx.Node (node_id, float(node_lon), float(node_lat))))])

        distances = map (float, edges_dict['LENGTH'])
        min_ph_dists = map (float, edges_dict['MINPHDISTOPEN'])
        edge_nodes = [(edges_dict['NODE0'][i], edges_dict['NODE1'][i]) for i in xrange(len(distances))]
        g.add_edges_from (edge_nodes)
        #min_ph_dists_all = 

        # Index for ph_dist weighting
        x = 1.

        for i, ns in enumerate (edge_nodes):
            g.edge[ns[0]][ns[1]]['distance'] = distances[i]
            g.edge[ns[0]][ns[1]]['ndist_div_phdist'] = distances[i] * 1.0 / (min_ph_dists[i])**(1./3.)
            g.edge[ns[0]][ns[1]]['phdist'] = min_ph_dists[i]
            g.edge[ns[0]][ns[1]]['inv_phdist'] = 1.0 / min_ph_dists[i]
            g.edge[ns[0]][ns[1]]['dist_phdist'] = distances[i] * min_ph_dists[i]



        # Find two nodes closest to pos0 and pos1 (by km dist)
        pos0_dists = {}
        pos1_dists = {}
        for n in g.nodes_iter():
            dist0 = geocalc (pos0[0], pos0[1],
                            g.node[n]['data'].lat, g.node[n]['data'].lon)
            pos0_dists[n] = dist0
            dist1 = geocalc (pos1[0], pos1[1],
                            g.node[n]['data'].lat, g.node[n]['data'].lon)
            pos1_dists[n] = dist1

        pos0_dists_df = pd.DataFrame (list(pos0_dists.iteritems()), 
                                         columns=['node', 'dist'])
        pos0_nearnode = pos0_dists_df[pos0_dists_df['dist'].isin([pos0_dists_df['dist'].min()])]

        pos1_dists_df = pd.DataFrame (list(pos1_dists.iteritems()), 
                                         columns=['node', 'dist'])
        pos1_nearnode = pos1_dists_df[pos1_dists_df['dist'].isin([pos1_dists_df['dist'].min()])]

        # Add start and dest nodes
        g.add_nodes_from ([('start0', dict (data=get_osm_nx.Node ('start0', pos0[1], pos0[0]))), 
                                ('dest0', dict (data=get_osm_nx.Node ('dest0', pos1[1], pos1[0])))])

        g.add_edge ('start0', str(pos0_nearnode['node'].values[0]))
        g.add_edge ('dest0', str(pos1_nearnode['node'].values[0]))



        # Get pothole data
        all_phs_df = misc.loading ('/var/www/RideSmoothly/app/static/baltimore_reported_phs_08_01_2014_08_05_2015_loc_openstreet.df')
        status = all_phs_df['status'].str.lower ()
        open_phs_withnans = all_phs_df[-status.str.contains ('clos')]
        open_phs = open_phs_withnans[open_phs_withnans.failed == 0]
        open_ph_addresses = open_phs['address_complete'].values
        open_ph_dates = open_phs['createdDate'].values
        open_phs_dict = {}
        ph_ids = get_ph_ids (open_phs['servicerequestnum'].values)
        ph_lats = open_phs['latitude'].values
        ph_lons = open_phs['longitude'].values
        for i in np.arange (len (ph_ids)):
            ph_id = ph_ids[i]
            lat = ph_lats[i]
            lon = ph_lons[i]
            open_phs_dict[ph_id] = (lat, lon)

        all_phs = all_phs_df[all_phs_df.failed == 0]
        ph_lats_all = all_phs['latitude'].values
        ph_lons_all = all_phs['longitude'].values


        # Add edge weights for start and dest edges
        n0s_bound = np.array (['start0', str(pos1_nearnode['node'].values[0])])
        n1s_bound = np.array ([str(pos0_nearnode['node'].values[0]), 'dest0'])
        lats0_bound, lons0_bound, lats1_bound, lons1_bound = get_path_coords_vect (g, n0s_bound, n1s_bound)
        distances_bound = geocalc_vect (lats0_bound, lons0_bound, lats1_bound, lons1_bound)


        f2 = np.vectorize (curry_geocalc_min (ph_lats, ph_lons))
        ph_dist_mins0_bound = f2 (lats0_bound, lons0_bound)
        ph_dist_mins1_bound = f2 (lats1_bound, lons1_bound)
        min_ph_dists_bound = np.min ([ph_dist_mins0_bound, ph_dist_mins1_bound], axis=0)


        g.edge['start0'][str(pos0_nearnode['node'].values[0])]['distance'] = distances_bound[0]
        g.edge['dest0'][str(pos1_nearnode['node'].values[0])]['distance'] = distances_bound[1]
        g.edge['start0'][str(pos0_nearnode['node'].values[0])]['ndist_div_phdist'] = distances_bound[0] * 1.0 / (min_ph_dists_bound[0])**(1./3.)
        g.edge['dest0'][str(pos1_nearnode['node'].values[0])]['ndist_div_phdist'] = distances_bound[1] * 1.0 / (min_ph_dists_bound[1])**(1./3.)
        g.edge['start0'][str(pos0_nearnode['node'].values[0])]['phdist'] = min_ph_dists_bound[0]
        g.edge['dest0'][str(pos1_nearnode['node'].values[0])]['phdist'] = min_ph_dists_bound[1]
        g.edge['start0'][str(pos0_nearnode['node'].values[0])]['inv_phdist'] = 1.0 / min_ph_dists_bound[0]
        g.edge['dest0'][str(pos1_nearnode['node'].values[0])]['inv_phdist'] = 1.0 / min_ph_dists_bound[1]
        g.edge['start0'][str(pos0_nearnode['node'].values[0])]['dist_phdist'] = distances_bound[0] * min_ph_dists_bound[0]
        g.edge['dest0'][str(pos1_nearnode['node'].values[0])]['dist_phdist'] = distances_bound[1] * min_ph_dists_bound[1]

        nodes_add = np.array(g.nodes())
        start_i = np.where (nodes_add == 'start0')[0][0]
        dest_i = np.where (nodes_add == 'dest0')[0][0]

        # Calculate shortest path weighted by distance
        shortest_path_dist = nx.shortest_path (g,
                                 source=nodes_add[start_i],
                                 target=nodes_add[dest_i],
                                 weight='distance')

        shortest_path_min_ph_dist_list = []
        for i in np.arange (0, len(shortest_path_dist) - 1):
            edge = g.edge[shortest_path_dist[i]][shortest_path_dist[i+1]]
            shortest_path_min_ph_dist_list.append (edge['phdist'])
        shortest_path_min_ph_dist = min(shortest_path_min_ph_dist_list)
        print ('shortest_path_min_ph_dist: {0}'.format (shortest_path_min_ph_dist))



        # Test if additional distance traveled is below ddist threshold
        shortest_roads = pd.DataFrame([g.edge[shortest_path_dist[i]][shortest_path_dist[i + 1]] 
                                       for i in range(len(shortest_path_dist) - 1)], 
                             columns=['distance', 'phdist', 'ndist_div_phdist'])

        shortest_distance = shortest_roads['distance'].sum ()


        path = nx.shortest_path (g,
                                 source=nodes_add[start_i],
                                 target=nodes_add[dest_i],
                                 weight='ndist_div_phdist')

        roads = pd.DataFrame([g.edge[path[i]][path[i + 1]] 
                              for i in range(len(path) - 1)], 
                             columns=['distance', 'phdist', 'ndist_div_phdist'])
        distance = roads['distance'].sum ()


        # Pothole tracking path
        path_phdist = nx.shortest_path (g,
                             source=nodes_add[start_i],
                             target=nodes_add[dest_i],
                             weight='dist_phdist')

        linepath_phdist = get_full_path (g, path_phdist)
        linepath_phdists = []
        linepath_phdists.append (linepath_phdist)

        linepaths = []
        short_not_smooth = []
        distancekm = []
        distancekm_shortest = [shortest_distance]


        def route_finder (distance, shortest_distance, x, ddist):
            print ('x = {0}'.format (x))
            print ('distance: {0}'.format (distance))
            print ('shortest_distance: {0}'.format (shortest_distance))
            print ('diff: {0}'.format (distance - shortest_distance))
            if  (distance - shortest_distance > ddist) and (distance != shortest_distance):
                x += 1.

                # get list with full graph, including start and dest nodes
                distances = nx.get_edge_attributes (g, 'distance').values ()
                min_ph_dists = nx.get_edge_attributes (g, 'phdist').values ()
                edges = nx.get_edge_attributes (g, 'distance').keys ()

                for i, ns in enumerate (edges):
                    g.edge[edges[i][0]][edges[i][1]]['ndist_div_phdist'] = distances[i] * 1.0 / (min_ph_dists[i])**(1./x)

                path = nx.shortest_path (g,
                                         source=nodes_add[start_i],
                                         target=nodes_add[dest_i],
                                         weight='ndist_div_phdist')

                roads = pd.DataFrame([g.edge[path[i]][path[i + 1]] 
                                      for i in range(len(path) - 1)], 
                                     columns=['distance', 'phdist', 'ndist_div_phdist'])
                distance = roads['distance'].sum ()
                print ('Smoothest Path: Total distance of route: {0} km, {1} mi'.format 
                       (roads['distance'].sum (), roads['distance'].sum () * 1./1.609))
                print ('Smoothest Path: Min pothole distance: {0} km, {1}'.format 
                       (min(roads['phdist']), min(roads['phdist']) * 1./1.609))
                print ('Smoothest Path: Total pothole distance: {0} km, {1}'.format 
                       (roads['phdist'].sum(), roads['phdist'].sum() * 1./1.609))
                print ('Smoothest Path: ndist_div_phdist: {0}'.format 
                       (roads['ndist_div_phdist'].sum()))

                route_finder (distance, shortest_distance, x, ddist)

            else:
                print ('Hit shortest and rough route! (or too close to short pothole route)')
                x -= 1.
                print ('Plotting previous iteration: the smoothest route! with x={0}'.format (x))

                # get list with full graph, including start and dest nodes
                distances = nx.get_edge_attributes (g, 'distance').values ()
                min_ph_dists = nx.get_edge_attributes (g, 'phdist').values ()
                edges = nx.get_edge_attributes (g, 'distance').keys ()

                for i, ns in enumerate (edges):
                    g.edge[edges[i][0]][edges[i][1]]['ndist_div_phdist'] = distances[i] * 1.0 / (min_ph_dists[i])**(1./x)

                path = nx.shortest_path (g,
                                         source=nodes_add[start_i],
                                         target=nodes_add[dest_i],
                                         weight='ndist_div_phdist')

                roads = pd.DataFrame([g.edge[path[i]][path[i + 1]] 
                                      for i in range(len(path) - 1)], 
                                     columns=['distance', 'phdist', 'ndist_div_phdist'])
                distance = roads['distance'].sum ()
                distancekm.append (distance)

                # Plot shortest route in red, then smoothest in green
                linepath = get_full_path (g, shortest_path_dist)
                short_not_smooth.append (linepath)

                linepath = get_full_path (g, path)
                linepaths.append (linepath)


        if (distance == shortest_distance) or (shortest_path_min_ph_dist > min_dph):
            print ('Smoothest route is shortest route!')
            # Plot smoothest and shortest route in green
            linepath = get_full_path (g, shortest_path_dist)
            linepaths.append (linepath)
            short_not_smooth.append (None)
            distancekm.append (distancekm_shortest[0])

        else:
            route_finder (distance, shortest_distance, x, ddist)


        
        ph_info = []
        for lat, lon, address, date in np.array ([ph_lats, ph_lons, open_ph_addresses, open_ph_dates]).T:
            ph_info.append ((lat, lon, address, date[:22]))

        ph_info_all = []
        for lat, lon in np.array ([ph_lats_all, ph_lons_all]).T:
            ph_info_all.append ([lat, lon])
                
        ## nparray.to_list() -> for javascript to use!
                
        closed_phs_withnans = all_phs_df[status.str.contains ('clos')]
        closed_phs = closed_phs_withnans[closed_phs_withnans.failed == 0]
        closed_ph_addresses = closed_phs['address_complete'].values
        closed_ph_dates = closed_phs['createdDate'].values
        closed_ph_statdate = closed_phs['statusDate'].values
        closed_ph_lats = closed_phs['latitude'].values
        closed_ph_lons = closed_phs['longitude'].values

        date_format = '%m/%d/%Y %I:%M:%S %p +0000'
        t_reported = get_dt_obj (closed_ph_dates, date_format)
        t_closed = get_dt_obj (closed_ph_statdate, date_format)
        dt = t_closed - t_reported
        dt_hours = get_dt_hours (dt)
        dt_days = get_dt_days (dt)
        ph_info_dt = []
        # for lat, lon, hours in np.array ([closed_ph_lats, closed_ph_lons, dt_hours]).T:
        #     ph_info_dt.append ([lat, lon, hours])
        for lat, lon, days in np.array ([closed_ph_lats, closed_ph_lons, dt_days]).T:
            ph_info_dt.append ([lat, lon, days])



        return start_loc, dest_loc, distancekm, distancekm_shortest, linepaths, short_not_smooth, linepath_phdists, ph_info, ph_info_all, centerlat, centerlon, ph_info_dt

