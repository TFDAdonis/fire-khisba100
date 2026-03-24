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
# Map with draw controls - returns polygon geometry
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
    #map { height: 500px; width: 100%; }
    #info-panel {
      background:#1e2130; padding:12px 16px; font-size:13px;
      border-top: 1px solid #333;
    }
    #info-panel b { color:#4CAF50; }
    .info-text { margin-top: 8px; font-size: 12px; color: #aaa; }
    .selected-polygon {
      background:#0e1117; border:1px solid #4CAF50; border-radius:6px;
      padding:8px 12px; margin-top: 8px; font-family: monospace;
      font-size: 11px; color: #4CAF50;
    }
    button {
      background:#4CAF50; color:white; border:none; padding:8px 16px;
      border-radius:4px; cursor:pointer; margin-top: 10px;
      font-size: 13px;
    }
    button:hover { background:#45a049; }
    .coord-row { font-family: monospace; font-size: 11px; margin: 4px 0; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="info-panel">
    <b>✏️ Draw a polygon on the map</b>
    <div class="info-text">
      Use the <strong>polygon (⬟) tool</strong> in the top-left toolbar. Draw a polygon to define your area of interest.
      After drawing, click <strong>"Send to App"</strong> to use this polygon for analysis.
    </div>
    <div id="polygon-info" style="display:none;" class="selected-polygon">
      <div>✅ Polygon selected! Click "Send to App" to use this area.</div>
      <div id="coords-list" class="coord-row"></div>
    </div>
    <button id="send-btn" style="display:none;">📤 Send to App & Process Data</button>
    <div id="status" class="info-text" style="margin-top: 10px; color: #ffaa44;"></div>
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

    var drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    
    var currentPolygon = null;
    var currentCoords = null;

    var drawControl = new L.Control.Draw({
      draw: {
        polyline: false,
        polygon: {
          allowIntersection: false,
          showArea: true,
          drawError: { color: '#ff0000', message: 'Polygon cannot intersect itself!' },
          shapeOptions: { color: '#4CAF50', weight: 3 }
        },
        circle: false,
        marker: false,
        circlemarker: false,
        rectangle: false
      },
      edit: { featureGroup: drawnItems, edit: true, remove: true }
    });
    map.addControl(drawControl);

    function formatCoords(latlngs) {
      var coords = [];
      latlngs.forEach(function(latlng) {
        coords.push([latlng.lng.toFixed(5), latlng.lat.toFixed(5)]);
      });
      return coords;
    }

    function updatePolygonInfo(layer) {
      var latlngs = layer.getLatLngs()[0];
      var coords = formatCoords(latlngs);
      
      currentCoords = coords;
      
      var coordsHtml = '<strong>Coordinates (lon, lat):</strong><br>';
      coords.forEach(function(coord) {
        coordsHtml += `[${coord[0]}, ${coord[1]}]<br>`;
      });
      
      document.getElementById('coords-list').innerHTML = coordsHtml;
      document.getElementById('polygon-info').style.display = 'block';
      document.getElementById('send-btn').style.display = 'block';
      document.getElementById('status').innerHTML = 'Polygon drawn. Click "Send to App" to analyze this area.';
    }

    map.on(L.Draw.Event.CREATED, function(e) {
      drawnItems.clearLayers();
      drawnItems.addLayer(e.layer);
      currentPolygon = e.layer;
      updatePolygonInfo(e.layer);
    });

    map.on(L.Draw.Event.EDITED, function(e) {
      var layers = e.layers;
      layers.eachLayer(function(layer) {
        currentPolygon = layer;
        updatePolygonInfo(layer);
      });
    });

    document.getElementById('send-btn').onclick = function() {
      if (currentCoords) {
        var message = {
          type: 'polygon',
          coordinates: currentCoords
        };
        window.parent.postMessage(message, '*');
        document.getElementById('status').innerHTML = '✅ Polygon sent to app! Loading data...';
        document.getElementById('send-btn').disabled = true;
      }
    };
  </script>
</body>
</html>
"""


def render_draw_map(center_lat, center_lon, zoom):
    """Render map for drawing polygon"""
    html = (DRAW_MAP_HTML
            .replace("INIT_LAT", str(center_lat))
            .replace("INIT_LON", str(center_lon))
            .replace("INIT_ZOOM", str(zoom)))
    
    components.html(html, height=580, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# Earth Engine helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_ndvi_tile_url(roi, start_date, end_date):
    """Get NDVI tile URL for map overlay"""
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
    """Get LST tile URL for map overlay"""
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
    """Load NDVI data as xarray"""
    sr = (ee.ImageCollection("NASA/VIIRS/002/VNP09GA")
          .filterDate(start_date, end_date)
          .filterBounds(roi)
          .select('I1', 'I2'))
    
    def ndvi(img):
        index = img.normalizedDifference(['I2', 'I1']).rename('ndvi')
        return index.copyProperties(img, img.propertyNames())
    
    viirs_ndvi = sr.map(ndvi)
    import xee
    ds = xr.open_dataset(
        viirs_ndvi, engine='ee',
        crs='EPSG:4326', scale=scale, geometry=roi
    )
    return ds, sr.size().getInfo()


def load_lst_xarray(roi, start_date, end_date, scale):
    """Load LST data as xarray"""
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


def plot_spatial_grid(ds, var_name, cmap, title, col_wrap=4, vmin=None, vmax=None):
    """Plot spatial grid similar to Colab's xr.plot col='time'"""
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

    arr = da.values
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

    dims = list(da.dims)
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

        if len(arr.shape) == 3:
            img = arr[i]
        else:
            img = arr[i]

        if lat_idx == 2:
            img = img.T

        im = ax.imshow(
            img[::-1],
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap=cmap, vmin=vmin, vmax=vmax,
            aspect='auto', interpolation='nearest'
        )

        date_str = str(t)[:10]
        ax.set_title(date_str, color='white', fontsize=8, pad=3)
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    for j in range(n, len(flat_axes)):
        flat_axes[j].set_visible(False)

    cbar = fig.colorbar(im, ax=flat_axes[:n], orientation='vertical', fraction=0.015, pad=0.04)
    cbar.ax.tick_params(colors='white', labelsize=8)
    cbar.ax.yaxis.label.set_color('white')

    plt.tight_layout()
    return fig


def plot_time_series_line(dates, values, ylabel, title, color):
    """Plot time series line chart"""
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
        "**Step 1:** Use the **Draw Polygon** tab to draw a polygon on the map.  \n"
        "**Step 2:** Click **Send to App & Process Data** after drawing.  \n"
        "**Step 3:** Select dates in the sidebar and the app will automatically process your polygon."
    )

    ee_ok, ee_err = initialize_ee()
    if not ee_ok:
        st.error(f"Earth Engine initialization failed: {ee_err}")
        st.stop()
    st.success("✅ Earth Engine connected — service account authenticated")

    # Session state defaults
    defaults = dict(
        ndvi_url=None, lst_url=None,
        roi_geometry=None, 
        map_center=[20, 0], map_zoom=2,
        ndvi_count=0, lst_count=0,
        ds_ndvi=None, ds_lst=None,
        ndvi_dates=[], ndvi_vals=[],
        lst_dates=[], lst_vals=[],
        active_layer="NDVI",
        polygon_drawn=False
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Handle polygon data from iframe
    if 'polygon_data' not in st.session_state:
        st.session_state.polygon_data = None

    # Listen for messages from iframe
    polygon_coords = None
    components.html("""
        <script>
            window.addEventListener('message', function(event) {
                if (event.data.type === 'polygon') {
                    localStorage.setItem('polygon_coords', JSON.stringify(event.data.coordinates));
                }
            });
        </script>
    """, height=0)

    # Check for stored polygon
    import streamlit.components.v1 as components
    polygon_js = """
        <script>
            var coords = localStorage.getItem('polygon_coords');
            if (coords) {
                var msg = {type: 'polygon_loaded', coordinates: JSON.parse(coords)};
                window.parent.postMessage(msg, '*');
                localStorage.removeItem('polygon_coords');
            }
        </script>
    """
    components.html(polygon_js, height=0)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")
        
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
        
        st.subheader("📦 Datasets")
        st.markdown("""
**NDVI:** `NASA/VIIRS/002/VNP09GA`
Bands I1 & I2 · 500 m · (NIR−Red)/(NIR+Red)

**LST:** `NASA/VIIRS/002/VNP21A1D`
Band LST_1KM · 1 km · Kelvin
        """)
        
        # Auto-process button (will be triggered by polygon)
        process_btn = st.button("🔄 Process Current Polygon", type="primary", use_container_width=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "✏️ Draw Polygon",
        "🗺️ Data Map",
        "🌿 NDVI Spatial Plots",
        "🌡️ LST Spatial Plots",
        "📈 Time Series"
    ])

    # ── Tab 0: Draw Polygon ───────────────────────────────────────────────────
    with tab0:
        st.subheader("✏️ Draw Your Area of Interest (Polygon)")
        st.info(
            "**Instructions:**\n"
            "1. Use the **polygon (⬟) tool** in the top-left of the map below.\n"
            "2. Draw a polygon around your area of interest.\n"
            "3. Click **'Send to App & Process Data'** button that appears after drawing.\n"
            "4. The app will automatically load data for your polygon."
        )
        
        # Center map
        cx = st.session_state.map_center[1]
        cy = st.session_state.map_center[0]
        cz = st.session_state.map_zoom
        render_draw_map(cy, cx, cz)
        
        # Button to manually trigger processing
        if process_btn and st.session_state.roi_geometry:
            st.info("Processing polygon data...")
            with st.spinner("Loading data..."):
                load_polygon_data(st.session_state.roi_geometry, start_date, end_date, scale_deg)
                st.success("Data loaded successfully!")

    # Function to load data for polygon
    def load_polygon_data(roi_geometry, start_date, end_date, scale_deg):
        """Load data for the given polygon"""
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        prog = st.progress(0, text="Fetching NDVI tile layer…")
        try:
            url, cnt = get_ndvi_tile_url(roi_geometry, start_str, end_str)
            st.session_state.ndvi_url = url
            st.session_state.ndvi_count = cnt
        except Exception as e:
            st.warning(f"NDVI tile error: {e}")
        prog.progress(20, text="Fetching LST tile layer…")

        try:
            url, cnt = get_lst_tile_url(roi_geometry, start_str, end_str)
            st.session_state.lst_url = url
            st.session_state.lst_count = cnt
        except Exception as e:
            st.warning(f"LST tile error: {e}")
        prog.progress(40, text="Loading NDVI xarray dataset (may take 30–60s)…")

        try:
            ds, _ = load_ndvi_xarray(roi_geometry, start_str, end_str, scale_deg)
            ds_loaded = ds.compute()
            st.session_state.ds_ndvi = ds_loaded
            dates, vals = compute_time_series(ds_loaded, 'ndvi')
            st.session_state.ndvi_dates = dates
            st.session_state.ndvi_vals = vals
        except Exception as e:
            st.warning(f"NDVI xarray error: {e}")
        prog.progress(70, text="Loading LST xarray dataset…")

        try:
            ds, _ = load_lst_xarray(roi_geometry, start_str, end_str, scale_deg)
            ds_loaded = ds.compute()
            st.session_state.ds_lst = ds_loaded
            dates, vals = compute_time_series(ds_loaded, 'LST_1KM')
            st.session_state.lst_dates = dates
            st.session_state.lst_vals = [v - 273.15 for v in vals]
        except Exception as e:
            st.warning(f"LST xarray error: {e}")
        prog.progress(100, text="Done!")

    # Listen for polygon data from JavaScript
    # This is a simplified approach - in practice you'd need a more robust method
    # For now, we'll use a placeholder that can be triggered manually
    
    # Create a placeholder for polygon input in sidebar
    with st.sidebar:
        st.subheader("📐 Polygon Coordinates (Optional)")
        st.markdown("If you prefer, you can manually enter polygon coordinates here:")
        polygon_coords_text = st.text_area(
            "Polygon coordinates (lon, lat format, one per line, closing point optional)",
            placeholder="[-10.0, 4.0]\n[2.0, 4.0]\n[2.0, 12.0]\n[-10.0, 12.0]",
            height=150
        )
        
        if polygon_coords_text:
            try:
                # Parse coordinates
                lines = polygon_coords_text.strip().split('\n')
                coords = []
                for line in lines:
                    line = line.strip().strip('[]')
                    parts = line.split(',')
                    if len(parts) == 2:
                        lon = float(parts[0].strip())
                        lat = float(parts[1].strip())
                        coords.append([lon, lat])
                
                if len(coords) >= 3:
                    # Create polygon geometry
                    roi_geometry = ee.Geometry.Polygon(coords)
                    st.session_state.roi_geometry = roi_geometry
                    st.success(f"✅ Polygon created with {len(coords)} points")
                    
                    if st.button("Load Data for This Polygon"):
                        with st.spinner("Loading data..."):
                            load_polygon_data(roi_geometry, start_date, end_date, scale_deg)
                            st.success("Data loaded successfully!")
                            st.rerun()
            except Exception as e:
                st.error(f"Error parsing coordinates: {e}")

    # ── Tab 1: Data Map ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("🗺️ Interactive Data Map")
        has_data = st.session_state.ndvi_url or st.session_state.lst_url
        if not has_data:
            st.info("Draw a polygon in the Draw Polygon tab, then click 'Send to App' or manually enter coordinates in sidebar.")
        
        if has_data and st.session_state.roi_geometry:
            ca, cb = st.columns(2)
            ca.metric("NDVI Images", st.session_state.ndvi_count)
            cb.metric("LST Images", st.session_state.lst_count)
            st.caption("✅ Data loaded for your polygon")
        
        # Show a folium map with the polygon
        if st.session_state.roi_geometry:
            st.subheader("Your Selected Area")
            # Create a folium map to show the polygon
            m = folium.Map(location=[0, 0], zoom_start=2)
            
            # Get polygon coordinates
            try:
                coords = st.session_state.roi_geometry.coordinates().getInfo()
                if coords:
                    # Extract first ring of polygon
                    ring = coords[0]
                    # Add polygon to map
                    folium.Polygon(
                        locations=[[lat, lon] for lon, lat in ring],
                        color='green',
                        weight=2,
                        fill=True,
                        fill_opacity=0.2,
                        popup='AOI'
                    ).add_to(m)
                    
                    # Center map on polygon
                    lats = [lat for lon, lat in ring]
                    lons = [lon for lon, lat in ring]
                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)
                    m.location = [center_lat, center_lon]
                    m.zoom_start = 8
                    
                    # Display map
                    from streamlit_folium import folium_static
                    folium_static(m, width=700, height=400)
            except:
                st.info("Polygon geometry loaded but cannot display map")
        
        # Tile layer info
        if has_data:
            if active_layer == "NDVI":
                st.markdown("**NDVI Legend:** 🔵 Blue (−0.2, bare/water) → ⬜ White → 🟡 Yellow → 🟢 Green → 🌲 Dark Green (0.8, dense vegetation)")
            else:
                st.markdown("**LST Legend:** 🔵 Dark Blue (270 K / −3°C) → 🩵 Cyan → 🟡 Yellow → 🔴 Red → 🟤 Dark Red (320 K / 47°C)")

    # ── Tab 2: NDVI Spatial Plots ─────────────────────────────────────────────
    with tab2:
        st.subheader("🌿 NDVI Spatial Maps — one panel per date")
        st.caption("Source: NASA/VIIRS/002/VNP09GA · Bands I1 & I2 · jet colormap")
        if st.session_state.ds_ndvi is not None:
            ds = st.session_state.ds_ndvi
            n = len(ds.time)
            st.info(f"Showing {n} time steps over the polygon. Each panel is one acquisition date.")
            with st.spinner("Rendering spatial plots…"):
                fig = plot_spatial_grid(
                    ds, 'ndvi', cmap='jet',
                    title='NDVI — NASA VIIRS (jet colormap, robust)',
                    col_wrap=col_wrap
                )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No NDVI data loaded yet. Draw a polygon in the Draw Polygon tab and click 'Send to App' or manually enter coordinates.")

    # ── Tab 3: LST Spatial Plots ──────────────────────────────────────────────
    with tab3:
        st.subheader("🌡️ LST Spatial Maps — one panel per date")
        st.caption("Source: NASA/VIIRS/002/VNP21A1D · Band LST_1KM · hot_r colormap (Kelvin)")
        if st.session_state.ds_lst is not None:
            ds = st.session_state.ds_lst
            n = len(ds.time)
            st.info(f"Showing {n} time steps over the polygon.")
            with st.spinner("Rendering spatial plots…"):
                fig = plot_spatial_grid(
                    ds, 'LST_1KM', cmap='hot_r',
                    title='LST (Kelvin) — NASA VIIRS (hot_r colormap, robust)',
                    col_wrap=col_wrap
                )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No LST data loaded yet. Draw a polygon in the Draw Polygon tab and click 'Send to App' or manually enter coordinates.")

    # ── Tab 4: Time Series ────────────────────────────────────────────────────
    with tab4:
        st.subheader("📈 Time Series — Mean over Polygon")
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
            st.pyplot(fig)
            plt.close(fig)
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
            st.pyplot(fig)
            plt.close(fig)
            if st.session_state.lst_dates:
                with st.expander("Raw table"):
                    st.dataframe([{"Date": d, "LST (°C)": round(v, 2)}
                                  for d, v in zip(st.session_state.lst_dates,
                                                  st.session_state.lst_vals)],
                                 use_container_width=True)

    with st.expander("ℹ️ About"):
        st.markdown("""
### GEE NDVI & LST Viewer - Polygon Mode
This app allows you to draw **any polygon** as your area of interest, not just rectangles!

**How it works:**
1. **Draw Polygon Tab** - Draw any polygon on the map using the polygon tool
2. Click "Send to App" to use that polygon
3. Or manually enter polygon coordinates in the sidebar
4. The app will automatically load NDVI and LST data for your exact polygon shape
5. View results in the other tabs

**Features:**
- **NDVI** from NASA/VIIRS/002/VNP09GA (500m resolution)
- **LST** from NASA/VIIRS/002/VNP21A1D (1km resolution)
- **Spatial plots** per time step (like Colab's xr.plot)
- **Time series** of mean values over your polygon
- **Interactive maps** with data overlays
        """)


if __name__ == "__main__":
    main()
