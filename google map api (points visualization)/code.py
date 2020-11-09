def putMarkers(markers):
    with open('map.html', 'w') as myFile:
        myFile.write('''<!DOCTYPE html>
    <html>
        <head>
            <title>world wide Map</title>
            <meta name="viewport" content="initial-scale=1.0">
            <meta charset="utf-8">
            <style>
                #map {
                    height: 100%;
                }
                html, body {
                    height: 100%;
                    margin: 0;
                    padding: 0;
                }
            </style>
        </head>
        <body>
            <div id="map"></div>
            <script>
                var map;
                function initMap() {
                    map = new google.maps.Map(document.getElementById('map'), {
                        center: {lat: -4.0383, lng:  21.7587},
                        zoom: 3
                    });

                    //add marker function
                    function addMarker(coords){
                        var marker = new google.maps.Marker({
                            position:coords,
                            map:map
                        });
                    }''')
        for lat,lng in markers:
            myFile.write("addMarker({lat:"+str(lat)+",lng:"+str(lng)+"});")
        myFile.write(''' }</script>
            <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCOuYp0vJWiadNMX7Y1YQALz4j0VtsV8_A	
            &callback=initMap" async defer></script>
        </body>
    </html>''')

# take care if lat is north, put it positive, and if it's south, put it negative
# if long is east, put it positive, and if it's west, put it negative
putMarkers([(30.0444, 31.2357),(31.2001, 29.9187),(25.6872, 32.6396),(41.0082, 28.9784),(39.9042, 116.4074)])
