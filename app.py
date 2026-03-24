import streamlit as st
import ee
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import streamlit.components.v1 as components
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="GEE NDVI & LST Viewer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_ee():
    try:
        private_key = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDFQOtXKWE+7mEY
JUTNzx3h+QvvDCvZ2B6XZTofknuAFPW2LqAzZustznJJFkCmO3Nutct+W/iDQCG0
1DjOQcbcr/jWr+mnRLVOkUkQc/kzZ8zaMQqU8HpXjS1mdhpsrbUaRKoEgfo3I3Bp
dFcJ/caC7TSr8VkGnZcPEZyXVsj8dLSEzomdkX+mDlJlgCrNfu3Knu+If5lXh3Me
SKiMWsfMnasiv46oD4szBzg6HLgoplmNka4NiwfeM7qROYnCd+5conyG8oiU00Xe
zC2Ekzo2dWsCw4zIJD6IdAcvgdrqH63fCqDFmAjEBZ69h8fWrdnsq56dAIpt0ygl
P9ADiRbVAgMBAAECggEALO7AnTqBGy2AgxhMP8iYEUdiu0mtvIIxV8HYl2QOC2ta
3GzrE8J0PJs8J99wix1cSmIRkH9hUP6dHvy/0uYjZ1aTi84HHtH1LghE2UFdySKy
RJqqwyozaDmx15b8Jnj8Wdc91miIR6KkQvVcNVuwalcf6jIAWlQwGp/jqIq9nloN
eld6xNbEmacORz1qT+4/uxOE05mrrZHC4kIKtswi8Io4ExVe61VxXsXWSHrMCGz0
TiSGr2ORSlRWC/XCGCu7zFIJU/iw6BiNsxryk6rjqQrcAtmoFTFx0fWbjYkG1DDs
k/9Dov1gyx0OtEyX8beoaf0Skcej4zdfeuido2A1sQKBgQD4IrhFn50i4/pa9sk1
g7v1ypGTrVA3pfvj6c7nTgzj9oyJnlU3WJwCqLw1cTFiY84+ekYP15wo8xsu5VZd
YLzOKEg3B8g899Ge14vZVNd6cNfRyMk4clGrDwGnZ4OAQkdsT/AyaCGRIcyu9njA
xdmWa+6VPMG7U65f/656XGwkBQKBgQDLgVyRE2+r1XCY+tdtXtga9sQ4LoiYHzD3
eDHe056qmwk8jf1A1HekILnC1GyeaKkOUd4TEWhVBgQpsvtC4Z2zPXlWR8N7SwNu
SFAhy3OnHTZQgrRWFA8eBjeI0YoXmk5m6uMQ7McmDlFxxXenFi+qSl3Cu4aGGuOy
cfyWMbTwkQKBgAoKfaJznww2ZX8g1WuQ9R4xIEr1jHV0BglnALRjeCoRZAZ9nb0r
nMSOx27yMallmIb2s7cYZn1RuRvgs+n7bCh7gNCZRAUTkiv3VPVqdX3C6zjWAy6B
kcR2Sv7XNX8PL4y2f2XKyPDyiTHbT2+dkfyASZtIZh6KeFfyJMFW1BlxAoGAAeG6
V2UUnUQl/GQlZc+AtA8gFVzoym9PZppn66WNTAqO9U5izxyn1o6u6QxJzNUu6wD6
yrZYfqDFnRUYma+4Y5Xn71JOjm9NItHsW8Oj2CG/BNOQk1MwKJjqHovBeSJmIzF8
1AU8ei+btS+cQaFE45A4ebp+LfNFs7q2GTVwdOECgYEAtHkMqigOmZdR3QAcZTjL
3aeOMGVHB2pHYosTgslD9Yp+hyVHqSdyCplHzWB3d8roIecW4MEb0mDxlaTdZfmR
dtBYiTzMxLezHsRZ4KP4NtGAE3iTL1b6DXuoI84+H/HaQ1EB79+YV9ZTAabt1b7o
e5aU1RW6tlG8nzHHwK2FeyI=
-----END PRIVATE KEY-----"""

        service_account_info = {
            "type": "service_account",
            "project_id": "citric-hawk-457513-i6",
            "private_key_id": "8984179a69969591194d8f8097e48cd9789f5ea2",
            "private_key": private_key,
            "client_email": "cc-365@citric-hawk-457513-i6.iam.gserviceaccount.com",
            "client_id": "105264622264803277310",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/cc-365%40citric-hawk-457513-i6.iam.gserviceaccount.com",
            "universe_domain": "googleapis.com"
        }

        credentials = ee.ServiceAccountCredentials(
            service_account_info['client_email'],
            key_data=json.dumps(service_account_info)
        )
        ee.Initialize(credentials, project='citric-hawk-457513-i6')
        return True, None
    except Exception as e:
        return False, str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Map with draw controls (pure Leaflet, no streamlit-folium)
# ─────────────────────────────────────────────────────────────────────────────
DRAW_MAP_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css"/>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#0e1117; color:#fafafa; font-family:sans-serif; }
    #map { height: 420px; width: 100%; }
    #coords-panel {
      background:#1e2130; padding:12px 16px; font-size:13px;
      border-top: 1px solid #333;
    }
    #coords-panel b { color:#4CAF50; }
    #coord-values {
      display:grid; grid-template-columns:1fr;
      gap:8px; margin-top:8px;
    }
    .coord-box {
      background:#0e1117; border:1px solid #444; border-radius:6px;
      padding:6px 10px; font-size:12px; color:#ccc;
      word-break:break-all;
      font-family:monospace;
    }
    .coord-box span { color:#fff; font-weight:bold; font-size:14px; display:block; margin-bottom:5px; }
    #hint { color:#aaa; font-size:12px; margin-top:6px; }
    #apply-btn {
      background:#4CAF50; color:white; border:none; padding:8px 16px;
      border-radius:4px; cursor:pointer; margin-top:10px; width:100%;
      font-size:14px; font-weight:bold;
    }
    #apply-btn:hover { background:#45a049; }
    .success-msg { color:#4CAF50; margin-top:8px; font-size:12px; }
    .error-msg { color:#ff6b35; margin-top:8px; font-size:12px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="coords-panel">
    <b>✏️ Draw a polygon on the map</b>
    <div id="coord-values">
      <div class="coord-box">
        <span>📐 Polygon Coordinates (click Apply to save)</span>
        <div id="polygon-coords" style="max-height:150px; overflow:auto; font-size:11px;">No polygon drawn yet</div>
      </div>
    </div>
    <div id="hint">
      Use the polygon tool (⬟) in the top-left toolbar to draw your area of interest.<br>
      After drawing, click "Apply Polygon" to use it for analysis.
    </div>
    <button id="apply-btn">✅ Apply Polygon for Analysis</button>
    <div id="status-msg"></div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
  <script>
    var map = L.map('map').setView([INIT_LAT, INIT_LON], INIT_ZOOM);

    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Esri World Imagery', maxZoom: 18
    }).addTo(map);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: 'OpenStreetMap', maxZoom: 19
    });

    // Show existing polygon if any
    EXISTING_POLY

    // EE overlay tiles
    EE_TILES

    var drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    var drawControl = new L.Control.Draw({
      draw: {
        polyline: false, polygon: true, circle: false,
        marker: false, circlemarker: false, rectangle: false
      },
      edit: { featureGroup: drawnItems, edit: true, remove: true }
    });
    map.addControl(drawControl);

    var currentPolygon = null;
    var currentCoords = null;

    function formatCoords(coords) {
      var formatted = coords.map(function(coord) {
        return '[' + coord[0].toFixed(5) + ', ' + coord[1].toFixed(5) + ']';
      }).join(', ');
      return '[' + formatted + ']';
    }

    function updatePolygonInfo(layer) {
      if (!layer) return;
      
      var latlngs = layer.getLatLngs()[0];
      var coords = latlngs.map(function(latlng) {
        return [latlng.lng, latlng.lat];
      });
      // Close the polygon by adding first point at end if not already closed
      if (coords[0][0] !== coords[coords.length-1][0] || 
          coords[0][1] !== coords[coords.length-1][1]) {
        coords.push(coords[0]);
      }
      currentCoords = coords;
      
      var coordStr = formatCoords(coords);
      document.getElementById('polygon-coords').innerHTML = 
        '<pre style="margin:0; color:#aaa;">Polygon coordinates:\\n' + coordStr + '</pre>';
    }

    map.on(L.Draw.Event.CREATED, function(e) {
      drawnItems.clearLayers();
      drawnItems.addLayer(e.layer);
      currentPolygon = e.layer;
      updatePolygonInfo(e.layer);
      document.getElementById('hint').innerHTML = 
        '✅ Polygon drawn! Click "Apply Polygon" to use it for analysis.';
      document.getElementById('status-msg').innerHTML = '';
    });

    map.on(L.Draw.Event.EDITED, function(e) {
      var layers = e.layers;
      layers.eachLayer(function(layer) {
        currentPolygon = layer;
        updatePolygonInfo(layer);
      });
      document.getElementById('hint').innerHTML = 
        '✅ Polygon edited! Click "Apply Polygon" to update analysis.';
      document.getElementById('status-msg').innerHTML = '';
    });

    document.getElementById('apply-btn').onclick = function() {
      if (!currentCoords) {
        document.getElementById('status-msg').innerHTML = 
          '<div class="error-msg">⚠️ Please draw a polygon first!</div>';
        return;
      }
      
      // Create GeoJSON
      var geojson = {
        type: "Polygon",
        coordinates: [currentCoords]
      };
      
      // Send polygon data to Streamlit
      var data = {
        type: "polygon_applied",
        geojson: geojson
      };
      
      // Send message to Streamlit
      if (window.parent !== window) {
        window.parent.postMessage({
          type: "streamlit:setComponentValue",
          value: JSON.stringify(data)
        }, "*");
      }
      
      document.getElementById('status-msg').innerHTML = 
        '<div class="success-msg">✅ Polygon applied! Go to sidebar and click "Load & Process Data"</div>';
    };
  </script>
</body>
</html>
"""


def render_draw_map(center_lat, center_lon, zoom, roi_geometry=None, ee_tile_url=None, layer_name="Data"):
    existing = ""
    if roi_geometry:
        try:
            # Convert ee.Geometry to GeoJSON
            if isinstance(roi_geometry, ee.Geometry):
                geojson = roi_geometry.getInfo()
            else:
                geojson = roi_geometry
            
            if geojson and 'coordinates' in geojson:
                coords = geojson['coordinates'][0]
                # Create Leaflet polygon
                latlngs = [[coord[1], coord[0]] for coord in coords]
                existing = f"""
                var polygon = L.polygon({json.dumps(latlngs)}, {{
                    color: '#FF4444',
                    weight: 2,
                    fill: true,
                    fillOpacity: 0.2
                }}).addTo(map);
                polygon.bindTooltip('Current AOI');
                drawnItems.addLayer(polygon);
                """
        except:
            pass

    ee_tiles = ""
    if ee_tile_url:
        ee_tiles = (
            f"L.tileLayer('{ee_tile_url}', "
            f"{{attribution:'Google Earth Engine — {layer_name}', maxZoom:18, opacity:0.8}})"
            f".addTo(map);"
        )

    html = (DRAW_MAP_HTML
            .replace("INIT_LAT", str(center_lat))
            .replace("INIT_LON", str(center_lon))
            .replace("INIT_ZOOM", str(zoom))
            .replace("EXISTING_POLY", existing)
            .replace("EE_TILES", ee_tiles))

    components.html(html, height=540, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# Earth Engine helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_ndvi_tile_url(roi, start_date, end_date):
    sr = (ee.ImageCollection("NASA/VIIRS/002/VNP09GA")
          .filterDate(start_date, end_date)
          .filterBounds(roi)
          .select('I1', 'I2'))
    count = sr.size().getInfo()
    mean_ndvi = sr.map(lambda img: img.normalizedDifference(['I2', 'I1']).rename('ndvi')).mean()
    vis = {'min': -0.2, 'max': 0.8, 'palette': ['blue', 'white', 'yellow', 'green', 'darkgreen']}
    url = mean_ndvi.getMapId(vis)['tile_fetcher'].url_format
    return url, count


def get_lst_tile_url(roi, start_date, end_date):
    lst = (ee.ImageCollection("NASA/VIIRS/002/VNP21A1D")
           .select('LST_1KM')
           .filterDate(start_date, end_date)
           .filterBounds(roi))
    count = lst.size().getInfo()
    vis = {'min': 270, 'max': 320,
           'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000', '#800000']}
    url = lst.mean().getMapId(vis)['tile_fetcher'].url_format
    return url, count


def load_ndvi_xarray(roi, start_date, end_date, scale):
    sr = (ee.ImageCollection("NASA/VIIRS/002/VNP09GA")
          .filterDate(start_date, end_date)
          .filterBounds(roi)
          .select('I1', 'I2'))
    viirs_ndvi = sr.map(lambda img: img.normalizedDifference(['I2', 'I1']).rename('ndvi').copyProperties(img, img.propertyNames()))
    import xee
    ds = xr.open_dataset(
        viirs_ndvi, engine='ee',
        crs='EPSG:4326', scale=scale, geometry=roi
    )
    return ds, sr.size().getInfo()


def load_lst_xarray(roi, start_date, end_date, scale):
    lst = (ee.ImageCollection("NASA/VIIRS/002/VNP21A1D")
           .select('LST_1KM')
           .filterDate(start_date, end_date)
           .filterBounds(roi))
    import xee
    ds = xr.open_dataset(
        lst, engine='ee',
        crs='EPSG:4326', scale=scale, geometry=roi
    )
    return ds, lst.size().getInfo()


def compute_time_series(ds, var_name, offset=0.0):
    """Reduce spatial mean over time from an xarray Dataset."""
    da = ds[var_name]
    means = da.mean(dim=['lon', 'lat']).values
    times = [str(t)[:10] for t in da.time.values]
    values = [float(v) + offset for v in means if not np.isnan(v)]
    dates = [times[i] for i, v in enumerate(means) if not np.isnan(v)]
    return dates, values


# ─────────────────────────────────────────────────────────────────────────────
# Colab-style spatial grid plot (ds.var.plot col='time'))
# ─────────────────────────────────────────────────────────────────────────────
def plot_spatial_grid(ds, var_name, cmap, title, col_wrap=4, vmin=None, vmax=None):
    da = ds[var_name]
    times = da.time.values
    n = len(times)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    ncols = min(col_wrap, n)
    nrows = int(np.ceil(n / ncols))
    fig_w = ncols * 3.5
    fig_h = nrows * 3.2

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#0e1117')
    fig.suptitle(title, color='white', fontsize=13, y=1.01)

    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    flat_axes = axes.flatten()

    arr = da.values  # (time, lat, lon) or (time, lon, lat) depending on ee output
    # Determine robust vmin/vmax
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
        return fig

    if vmin is None:
        vmin = float(np.nanpercentile(valid, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(valid, 98))

    # Detect dimension order
    dims = list(da.dims)  # e.g. ('time','lat','lon') or ('time','lon','lat')
    if 'lat' in dims and 'lon' in dims:
        lat_idx = dims.index('lat')
        lon_idx = dims.index('lon')
        lats = da['lat'].values
        lons = da['lon'].values
    else:
        lat_idx, lon_idx = 1, 2
        lats = np.arange(arr.shape[1])
        lons = np.arange(arr.shape[2])

    for i, t in enumerate(times):
        ax = flat_axes[i]
        ax.set_facecolor('#0e1117')

        # slice this time step
        if len(arr.shape) == 3:
            img = arr[i]
        else:
            img = arr[i]

        # If dims are (time, lon, lat), transpose to (lat, lon)
        if lat_idx == 2:
            img = img.T

        im = ax.imshow(
            img[::-1],  # flip latitude axis so north is up
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=cmap, vmin=vmin, vmax=vmax,
            aspect='auto', interpolation='nearest'
        )

        date_str = str(t)[:10]
        ax.set_title(date_str, color='white', fontsize=8, pad=3)
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    # Hide unused axes
    for j in range(n, len(flat_axes)):
        flat_axes[j].set_visible(False)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=flat_axes[:n], orientation='vertical', fraction=0.015, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.ax.yaxis.label.set_color('white')

    plt.tight_layout()
    return fig


def plot_time_series_line(dates, values, ylabel, title, color):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#1e2130')
    ax.set_facecolor('#1e2130')
    if dates:
        x = list(range(len(dates)))
        ax.fill_between(x, values, alpha=0.25, color=color)
        ax.plot(x, values, 'o-', color=color, linewidth=2, markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha='right', color='white', fontsize=8)
        ax.grid(True, alpha=0.2, color='gray')
        ax.set_ylim(min(values) * 0.95, max(values) * 1.05)
    else:
        ax.text(0.5, 0.5, 'No data — load data first', ha='center', va='center',
                color='white', transform=ax.transAxes, fontsize=13)
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white', fontsize=12, pad=8)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#555')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("🌍 Google Earth Engine — NDVI & LST Viewer")
    st.markdown(
        "**Step 1:** Use the **Draw Map** tab to draw a polygon (⬟ tool) on the map.  \n"
        "**Step 2:** Click **Apply Polygon** after drawing.  \n"
        "**Step 3:** Select date range in sidebar and click **Load & Process Data**."
    )

    ee_ok, ee_err = initialize_ee()
    if not ee_ok:
        st.error(f"Earth Engine initialization failed: {ee_err}")
        st.stop()
    st.success("✅ Earth Engine connected — service account authenticated")

    # Session state defaults
    defaults = dict(
        ndvi_url=None, lst_url=None,
        roi_geometry=None,  # Store ee.Geometry object
        roi_geojson=None,   # Store GeoJSON for display
        map_center=[20, 0], map_zoom=2,
        ndvi_count=0, lst_count=0,
        ds_ndvi=None, ds_lst=None,
        ndvi_dates=[], ndvi_vals=[],
        lst_dates=[], lst_vals=[],
        active_layer="NDVI"
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Handle polygon data from query parameters
    query_params = st.query_params
    
    if 'polygon_geojson' in query_params:
        try:
            polygon_geojson = json.loads(query_params['polygon_geojson'])
            if polygon_geojson and 'coordinates' in polygon_geojson:
                st.session_state.roi_geojson = polygon_geojson
                # Create ee.Geometry from GeoJSON
                st.session_state.roi_geometry = ee.Geometry(polygon_geojson)
                st.success("✅ Polygon loaded from map!")
                st.rerun()
        except:
            pass
    
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        # Display current AOI status
        if st.session_state.roi_geojson:
            st.success("✅ AOI Polygon Defined")
            with st.expander("View Current AOI"):
                st.json(st.session_state.roi_geojson)
        else:
            st.warning("⚠️ No AOI defined yet. Draw a polygon in the Draw Map tab and click Apply Polygon!")
        
        st.subheader("📅 Date Range")
        start_date = st.date_input("Start Date", value=datetime.date(2025, 1, 1),
                                   min_value=datetime.date(2012, 1, 1),
                                   max_value=datetime.date.today())
        end_date = st.date_input("End Date", value=datetime.date(2025, 2, 1),
                                 min_value=datetime.date(2012, 1, 1),
                                 max_value=datetime.date.today())
        if start_date >= end_date:
            st.error("End date must be after start date.")
            st.stop()

        st.subheader("🗺️ Map Overlay")
        active_layer = st.radio("Show on map", ["NDVI", "LST"])

        st.subheader("🔬 Spatial Plot Scale")
        scale_deg = st.slider("Resolution (degrees)", min_value=0.005, max_value=0.1,
                              value=0.01, step=0.005, format="%.3f")
        st.caption(f"≈ {scale_deg * 111:.1f} km/pixel  (smaller = more detail, slower)")

        col_wrap = st.slider("Columns per row in grid plots", 2, 6, 4)

        st.divider()
        
        # Only enable button if AOI is defined
        run_btn = st.button("🚀 Load & Process Data", type="primary", 
                           use_container_width=True,
                           disabled=st.session_state.roi_geometry is None)
        
        if st.session_state.roi_geometry is None:
            st.info("📝 Draw a polygon in the Draw Map tab and click Apply Polygon first!")

        st.subheader("📦 Datasets")
        st.markdown("""
**NDVI:** `NASA/VIIRS/002/VNP09GA`
Bands I1 & I2 · 500 m · (NIR−Red)/(NIR+Red)

**LST:** `NASA/VIIRS/002/VNP21A1D`
Band LST_1KM · 1 km · Kelvin
        """)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🖊️ Draw Map",
        "🗺️ Data Map",
        "🌿 NDVI Spatial Plots",
        "🌡️ LST Spatial Plots",
        "📈 Time Series"
    ])

    # ── Process ───────────────────────────────────────────────────────────────
    if run_btn and st.session_state.roi_geometry:
        roi = st.session_state.roi_geometry
        start_str = start_date.strftime('%Y-%m-%d')
        end_str   = end_date.strftime('%Y-%m-%d')

        st.session_state.active_layer = active_layer
        
        # Get bounding box for map centering
        bounds = roi.bounds().getInfo()['coordinates'][0]
        lons = [coord[0] for coord in bounds]
        lats = [coord[1] for coord in bounds]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
        st.session_state.map_center = [center_lat, center_lon]
        st.session_state.map_zoom = 8
        
        st.session_state.ds_ndvi = None
        st.session_state.ds_lst = None

        prog = st.progress(0, text="Fetching NDVI tile layer…")
        try:
            url, cnt = get_ndvi_tile_url(roi, start_str, end_str)
            st.session_state.ndvi_url   = url
            st.session_state.ndvi_count = cnt
        except Exception as e:
            st.warning(f"NDVI tile error: {e}")
        prog.progress(20, text="Fetching LST tile layer…")

        try:
            url, cnt = get_lst_tile_url(roi, start_str, end_str)
            st.session_state.lst_url   = url
            st.session_state.lst_count = cnt
        except Exception as e:
            st.warning(f"LST tile error: {e}")
        prog.progress(40, text="Loading NDVI xarray dataset (may take 30–60s)…")

        try:
            ds, _ = load_ndvi_xarray(roi, start_str, end_str, scale_deg)
            # Force compute of a small preview to catch errors early
            ds_loaded = ds.compute()
            st.session_state.ds_ndvi = ds_loaded
            dates, vals = compute_time_series(ds_loaded, 'ndvi')
            st.session_state.ndvi_dates = dates
            st.session_state.ndvi_vals  = vals
        except Exception as e:
            st.warning(f"NDVI xarray error: {e}")
        prog.progress(70, text="Loading LST xarray dataset…")

        try:
            ds, _ = load_lst_xarray(roi, start_str, end_str, scale_deg)
            ds_loaded = ds.compute()
            st.session_state.ds_lst = ds_loaded
            dates, vals = compute_time_series(ds_loaded, 'LST_1KM')
            # Convert K → C
            st.session_state.lst_dates = dates
            st.session_state.lst_vals  = [v - 273.15 for v in vals]
        except Exception as e:
            st.warning(f"LST xarray error: {e}")
        prog.progress(100, text="Done!")
        st.success("✅ All data loaded — see tabs for results")
        st.rerun()

    # ── Tab 0: Draw Map ───────────────────────────────────────────────────────
    with tab0:
        st.subheader("🖊️ Draw Your Area of Interest")
        st.info(
            "1. Use the **polygon tool (⬟)** in the top-left toolbar to draw your area of interest.\n"
            "2. After drawing, click **Apply Polygon** to save it for analysis.\n"
            "3. Go to sidebar and select date range, then click **Load & Process Data**."
        )
        
        # Center map on current ROI or default
        if st.session_state.roi_geometry:
            bounds = st.session_state.roi_geometry.bounds().getInfo()['coordinates'][0]
            lons = [coord[0] for coord in bounds]
            lats = [coord[1] for coord in bounds]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            cx = center_lon
            cy = center_lat
            cz = 8
        else:
            cx = 0
            cy = 20
            cz = 2
        
        render_draw_map(cy, cx, cz, roi_geometry=st.session_state.roi_geometry)

    # ── Tab 1: Data Map ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("🗺️ Interactive Data Map")
        has_data = st.session_state.ndvi_url or st.session_state.lst_url
        if not has_data:
            st.info("Draw polygon in Draw Map tab, then click Load & Process Data.")
        
        if has_data and st.session_state.roi_geometry:
            ca, cb = st.columns(2)
            ca.metric("NDVI Images", st.session_state.ndvi_count)
            cb.metric("LST Images",  st.session_state.lst_count)
        
        if st.session_state.roi_geometry:
            tile_url = (st.session_state.ndvi_url if active_layer == "NDVI"
                        else st.session_state.lst_url)
            
            # Center map on ROI
            bounds = st.session_state.roi_geometry.bounds().getInfo()['coordinates'][0]
            lons = [coord[0] for coord in bounds]
            lats = [coord[1] for coord in bounds]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            cx = center_lon
            cy = center_lat
            cz = 8
            
            render_draw_map(cy, cx, cz,
                            roi_geometry=st.session_state.roi_geometry,
                            ee_tile_url=tile_url,
                            layer_name=active_layer)
            
            if has_data:
                if active_layer == "NDVI":
                    st.markdown("**NDVI Legend:** 🔵 Blue (−0.2, bare/water) → ⬜ White → 🟡 Yellow → 🟢 Green → 🌲 Dark Green (0.8, dense vegetation)")
                else:
                    st.markdown("**LST Legend:** 🔵 Dark Blue (270 K / −3°C) → 🩵 Cyan → 🟡 Yellow → 🔴 Red → 🟤 Dark Red (320 K / 47°C)")

    # ── Tab 2: NDVI Spatial Plots ─────────────────────────────────────────────
    with tab2:
        st.subheader("🌿 NDVI Spatial Maps — one panel per date (Colab-style)")
        st.caption("Source: NASA/VIIRS/002/VNP09GA · Bands I1 & I2 · jet colormap")
        if st.session_state.ds_ndvi is not None:
            ds = st.session_state.ds_ndvi
            n = len(ds.time)
            st.info(f"Showing {n} time steps over the AOI. Each panel is one acquisition date.")
            with st.spinner("Rendering spatial plots…"):
                fig = plot_spatial_grid(
                    ds, 'ndvi', cmap='jet',
                    title='NDVI — NASA VIIRS (jet colormap, robust)',
                    col_wrap=col_wrap
                )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No NDVI data loaded yet. Draw polygon in Draw Map tab then click Load & Process Data.")

    # ── Tab 3: LST Spatial Plots ──────────────────────────────────────────────
    with tab3:
        st.subheader("🌡️ LST Spatial Maps — one panel per date (Colab-style)")
        st.caption("Source: NASA/VIIRS/002/VNP21A1D · Band LST_1KM · hot_r colormap (Kelvin)")
        if st.session_state.ds_lst is not None:
            ds = st.session_state.ds_lst
            n = len(ds.time)
            st.info(f"Showing {n} time steps over the AOI.")
            with st.spinner("Rendering spatial plots…"):
                fig = plot_spatial_grid(
                    ds, 'LST_1KM', cmap='hot_r',
                    title='LST (Kelvin) — NASA VIIRS (hot_r colormap, robust)',
                    col_wrap=col_wrap
                )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No LST data loaded yet. Draw polygon in Draw Map tab then click Load & Process Data.")

    # ── Tab 4: Time Series ────────────────────────────────────────────────────
    with tab4:
        st.subheader("📈 Time Series — Mean over AOI")
        col_n, col_l = st.columns(2)

        with col_n:
            st.markdown("**NDVI** (mean)")
            if st.session_state.ndvi_dates:
                v = st.session_state.ndvi_vals
                st.metric("Mean NDVI", f"{sum(v)/len(v):.3f}")
            fig = plot_time_series_line(
                st.session_state.ndvi_dates, st.session_state.ndvi_vals,
                "NDVI", "NDVI over Time", "#4CAF50"
            )
            st.pyplot(fig); plt.close(fig)
            if st.session_state.ndvi_dates:
                with st.expander("Raw table"):
                    st.dataframe([{"Date": d, "NDVI": round(v, 4)}
                                  for d, v in zip(st.session_state.ndvi_dates,
                                                  st.session_state.ndvi_vals)],
                                 use_container_width=True)

        with col_l:
            st.markdown("**LST** (mean, °C)")
            if st.session_state.lst_dates:
                v = st.session_state.lst_vals
                st.metric("Mean LST", f"{sum(v)/len(v):.1f}°C")
            fig = plot_time_series_line(
                st.session_state.lst_dates, st.session_state.lst_vals,
                "Temperature (°C)", "LST over Time", "#FF6B35"
            )
            st.pyplot(fig); plt.close(fig)
            if st.session_state.lst_dates:
                with st.expander("Raw table"):
                    st.dataframe([{"Date": d, "LST (°C)": round(v, 2)}
                                  for d, v in zip(st.session_state.lst_dates,
                                                  st.session_state.lst_vals)],
                                 use_container_width=True)

    with st.expander("ℹ️ About"):
        st.markdown("""
### GEE NDVI & LST Viewer
Connects to Google Earth Engine via service account `citric-hawk-457513-i6`.

**Workflow:**
1. **Draw Map** — Draw a polygon on the map using the polygon tool (⬟)
2. **Apply Polygon** — Click "Apply Polygon" to save your AOI
3. **Load Data** — Select date range and click "Load & Process Data"

**Tabs:**
- **Draw Map** — draw polygon to define your area of interest  
- **Data Map** — interactive map with NDVI or LST overlay from EE tile server  
- **NDVI Spatial Plots** — grid of spatial NDVI maps per time step  
- **LST Spatial Plots** — same for land surface temperature  
- **Time Series** — line charts of spatial mean NDVI and LST over time  

**Note:** All analysis is performed only on the polygon you draw. No default geometry is used.
        """)


if __name__ == "__main__":
    main()
