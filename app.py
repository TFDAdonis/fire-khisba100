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
    #map { height: 400px; width: 100%; }
    #coords-panel {
      background:#1e2130; 
      padding: 15px;
      border-top: 2px solid #4CAF50;
      max-height: 200px;
      overflow-y: auto;
    }
    .panel-title {
      font-size: 14px;
      font-weight: bold;
      color: #4CAF50;
      margin-bottom: 12px;
      text-align: center;
    }
    .coord-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 12px;
    }
    .coord-card {
      flex: 1;
      min-width: 200px;
      background: #0e1117;
      border-radius: 8px;
      padding: 10px;
      border-left: 3px solid #4CAF50;
    }
    .coord-label {
      font-size: 11px;
      color: #888;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .coord-value {
      font-size: 20px;
      font-weight: bold;
      font-family: 'Courier New', monospace;
      color: #fff;
      margin: 5px 0;
    }
    .coord-desc {
      font-size: 10px;
      color: #666;
    }
    .hint-box {
      background: rgba(76, 175, 80, 0.1);
      border-radius: 6px;
      padding: 10px;
      text-align: center;
      font-size: 12px;
      color: #FFA500;
      margin-top: 10px;
    }
    .success-hint {
      background: rgba(76, 175, 80, 0.2);
      color: #4CAF50;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div id="coords-panel">
    <div class="panel-title">📐 DRAW RECTANGLE TO GET COORDINATES</div>
    <div class="coord-grid" id="coord-grid">
      <div class="coord-card">
        <div class="coord-label">WEST LONGITUDE</div>
        <div class="coord-value" id="min-lon">—</div>
        <div class="coord-desc">Min Lon (left boundary)</div>
      </div>
      <div class="coord-card">
        <div class="coord-label">EAST LONGITUDE</div>
        <div class="coord-value" id="max-lon">—</div>
        <div class="coord-desc">Max Lon (right boundary)</div>
      </div>
      <div class="coord-card">
        <div class="coord-label">SOUTH LATITUDE</div>
        <div class="coord-value" id="min-lat">—</div>
        <div class="coord-desc">Min Lat (bottom boundary)</div>
      </div>
      <div class="coord-card">
        <div class="coord-label">NORTH LATITUDE</div>
        <div class="coord-value" id="max-lat">—</div>
        <div class="coord-desc">Max Lat (top boundary)</div>
      </div>
    </div>
    <div class="hint-box" id="hint">
      ✏️ Click the ▭ rectangle tool in top-left corner, draw a rectangle on the map
    </div>
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

    // Show existing AOI if any
    EXISTING_RECT

    // EE overlay tiles
    EE_TILES

    var drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    var drawControl = new L.Control.Draw({
      draw: {
        polyline: false, polygon: false, circle: false,
        marker: false, circlemarker: false, rectangle: true
      },
      edit: { featureGroup: drawnItems, edit: false, remove: true }
    });
    map.addControl(drawControl);

    function fmt(n) { return n.toFixed(5); }

    map.on(L.Draw.Event.CREATED, function(e) {
      drawnItems.clearLayers();
      drawnItems.addLayer(e.layer);
      var b = e.layer.getBounds();
      var minLon = fmt(b.getWest());
      var maxLon = fmt(b.getEast());
      var minLat = fmt(b.getSouth());
      var maxLat = fmt(b.getNorth());
      
      document.getElementById('min-lon').textContent = minLon;
      document.getElementById('max-lon').textContent = maxLon;
      document.getElementById('min-lat').textContent = minLat;
      document.getElementById('max-lat').textContent = maxLat;
      
      var hintDiv = document.getElementById('hint');
      hintDiv.innerHTML = '✅ COORDINATES CAPTURED! Copy these values to sidebar inputs: Min Lon=' + minLon + ', Max Lon=' + maxLon + ', Min Lat=' + minLat + ', Max Lat=' + maxLat;
      hintDiv.className = 'hint-box success-hint';
    });
  </script>
</body>
</html>
"""


def render_draw_map(center_lat, center_lon, zoom, roi_coords=None, ee_tile_url=None, layer_name="Data"):
    existing = ""
    if roi_coords:
        mn_lon, mn_lat, mx_lon, mx_lat = roi_coords
        existing = (
            f"L.rectangle([[{mn_lat},{mn_lon}],[{mx_lat},{mx_lon}]], "
            f"{{color:'#FF4444', weight:2, fill:true, fillOpacity:0.1}})"
            f".addTo(map).bindTooltip('Current AOI');"
        )

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
            .replace("EXISTING_RECT", existing)
            .replace("EE_TILES", ee_tiles))

    components.html(html, height=620, scrolling=False)


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
        "**Step 1:** Use the **Draw Map** tab to draw a rectangle and get AOI coordinates.  \n"
        "**Step 2:** Enter those coordinates + dates in the sidebar.  \n"
        "**Step 3:** Click **Load & Process Data** to generate all visualizations."
    )

    ee_ok, ee_err = initialize_ee()
    if not ee_ok:
        st.error(f"Earth Engine initialization failed: {ee_err}")
        st.stop()
    st.success("✅ Earth Engine connected — service account authenticated")

    # Session state defaults
    defaults = dict(
        ndvi_url=None, lst_url=None,
        roi_coords=None, map_center=[20, 0], map_zoom=2,
        ndvi_count=0, lst_count=0,
        ds_ndvi=None, ds_lst=None,
        ndvi_dates=[], ndvi_vals=[],
        lst_dates=[], lst_vals=[],
        active_layer="NDVI"
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

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

        st.subheader("📐 AOI Bounding Box")
        st.caption("Draw on the map (tab 1) to find coordinates, then enter them here.")
        
        # Create a more intuitive layout with 2 columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Longitude**")
            min_lon = st.number_input("Min (West)", value=-10.0, min_value=-180.0, max_value=180.0, format="%.5f", step=0.5, key="min_lon")
            st.markdown("**Latitude**")
            min_lat = st.number_input("Min (South)", value=4.0, min_value=-90.0, max_value=90.0, format="%.5f", step=0.5, key="min_lat")
        with col2:
            st.markdown("**Longitude**")
            max_lon = st.number_input("Max (East)", value=2.0, min_value=-180.0, max_value=180.0, format="%.5f", step=0.5, key="max_lon")
            st.markdown("**Latitude**")
            max_lat = st.number_input("Max (North)", value=12.0, min_value=-90.0, max_value=90.0, format="%.5f", step=0.5, key="max_lat")

        if min_lon >= max_lon or min_lat >= max_lat:
            st.error("Min values must be less than Max values.")
            st.stop()

        st.subheader("🗺️ Map Overlay")
        active_layer = st.radio("Show on map", ["NDVI", "LST"])

        st.subheader("🔬 Spatial Plot Scale")
        scale_deg = st.slider("Resolution (degrees)", min_value=0.005, max_value=0.1,
                              value=0.01, step=0.005, format="%.3f")
        st.caption(f"≈ {scale_deg * 111:.1f} km/pixel  (smaller = more detail, slower)")

        col_wrap = st.slider("Columns per row in grid plots", 2, 6, 4)

        st.divider()
        run_btn = st.button("🚀 Load & Process Data", type="primary", use_container_width=True)

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
    if run_btn:
        roi_coords = (min_lon, min_lat, max_lon, max_lat)
        roi = ee.Geometry.Rectangle(list(roi_coords))
        start_str = start_date.strftime('%Y-%m-%d')
        end_str   = end_date.strftime('%Y-%m-%d')

        st.session_state.roi_coords   = roi_coords
        st.session_state.map_center   = [(min_lat + max_lat)/2, (min_lon + max_lon)/2]
        st.session_state.map_zoom     = 7
        st.session_state.active_layer = active_layer
        st.session_state.ds_ndvi      = None
        st.session_state.ds_lst       = None

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
        st.subheader("🖊️ Draw your Area of Interest")
        st.info(
            "Use the **rectangle (▭) tool** in the top-left of the map below. "
            "After drawing, the bounding box coordinates will appear in the panel below the map. "
            "Copy those values into the **AOI Bounding Box** inputs in the sidebar, then click **Load & Process Data**."
        )
        cx = st.session_state.map_center[1]
        cy = st.session_state.map_center[0]
        cz = st.session_state.map_zoom
        render_draw_map(cy, cx, cz, roi_coords=st.session_state.roi_coords)

    # ── Tab 1: Data Map ───────────────────────────────────────────────────────
    with tab1:
        st.subheader("🗺️ Interactive Data Map")
        has_data = st.session_state.ndvi_url or st.session_state.lst_url
        if not has_data:
            st.info("Enter AOI & dates in the sidebar, then click **Load & Process Data**.")

        if has_data:
            ca, cb = st.columns(2)
            ca.metric("NDVI Images", st.session_state.ndvi_count)
            cb.metric("LST Images",  st.session_state.lst_count)
            if st.session_state.roi_coords:
                rc = st.session_state.roi_coords
                st.caption(f"AOI: lon [{rc[0]:.3f} → {rc[2]:.3f}], lat [{rc[1]:.3f} → {rc[3]:.3f}]")

        tile_url = (st.session_state.ndvi_url if active_layer == "NDVI"
                    else st.session_state.lst_url)
        cx = st.session_state.map_center[1]
        cy = st.session_state.map_center[0]
        cz = st.session_state.map_zoom
        render_draw_map(cy, cx, cz,
                        roi_coords=st.session_state.roi_coords,
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
            st.info("No NDVI data loaded yet. Enter AOI & dates in sidebar then click **Load & Process Data**.")

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
            st.info("No LST data loaded yet. Enter AOI & dates in sidebar then click **Load & Process Data**.")

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

**Tabs:**
- **Draw Map** — draw a rectangle to discover bounding box coordinates  
- **Data Map** — interactive map with NDVI or LST overlay from EE tile server  
- **NDVI Spatial Plots** — grid of spatial NDVI maps per time step (like Colab `xr.plot col='time'`)  
- **LST Spatial Plots** — same for land surface temperature  
- **Time Series** — line charts of spatial mean NDVI and LST over time  
        """)


if __name__ == "__main__":
    main()
