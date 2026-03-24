import streamlit as st
import ee
import datetime
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import folium
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


def create_map():
    """Create a map with drawing controls"""
    m = geemap.Map(
        center=[20, 0],
        zoom=2,
        basemap='HYBRID',
        draw_ctrl=True,
        draw_export=True
    )
    
    # Add drawing control for rectangle only
    m.draw_control = {
        'draw': {
            'polyline': False,
            'polygon': False,
            'circle': False,
            'marker': False,
            'circlemarker': False,
            'rectangle': True
        }
    }
    
    return m


def get_ndvi_data(roi, start_date, end_date, scale=0.001):
    """Load NDVI data exactly like the original code"""
    sr = ee.ImageCollection("NASA/VIIRS/002/VNP09GA")\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)\
        .select('I1', 'I2')
    
    def ndvi(img):
        index = img.normalizedDifference(['I2','I1']).rename('ndvi')
        return index.copyProperties(img, img.propertyNames())
    
    viirs_ndvi = sr.map(ndvi)
    count = viirs_ndvi.size().getInfo()
    
    if count > 0:
        ds = xr.open_dataset(
            viirs_ndvi,
            engine='ee',
            crs='EPSG:4326',
            scale=scale,
            geometry=roi
        )
        return ds, count
    return None, 0


def get_lst_data(roi, start_date, end_date, scale=0.005):
    """Load LST data exactly like the original code"""
    lst = ee.ImageCollection("NASA/VIIRS/002/VNP21A1D")\
        .select('LST_1KM')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)
    
    count = lst.size().getInfo()
    
    if count > 0:
        ds = xr.open_dataset(
            lst,
            engine='ee',
            crs='EPSG:4326',
            scale=scale,
            geometry=roi
        )
        return ds, count
    return None, 0


def plot_ndvi_grid(ds):
    """Plot NDVI time series grid"""
    if ds is None:
        return None
    
    n = len(ds.time)
    if n == 0:
        return None
    
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    fig.suptitle('NDVI Time Series', fontsize=16, y=1.02)
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ds.ndvi.isel(time=i).plot(ax=ax, cmap='jet', robust=True)
        date_str = str(ds.time.values[i])[:10]
        ax.set_title(date_str)
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_lst_grid(ds):
    """Plot LST time series grid"""
    if ds is None:
        return None
    
    n = len(ds.time)
    if n == 0:
        return None
    
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows))
    fig.suptitle('LST Time Series (Kelvin)', fontsize=16, y=1.02)
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ds.LST_1KM.isel(time=i).plot(ax=ax, cmap='hot_r', robust=True)
        date_str = str(ds.time.values[i])[:10]
        ax.set_title(date_str)
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def main():
    st.title("🌍 Google Earth Engine — NDVI & LST Viewer")
    st.markdown("""
    **How to use:**
    1. **Draw a rectangle** on the map below (use the rectangle tool in the top-left)
    2. Select your **date range** in the sidebar
    3. Click **Load & Process Data** to analyze your area
    """)
    
    # Initialize EE
    ee_ok, ee_err = initialize_ee()
    if not ee_ok:
        st.error(f"Earth Engine initialization failed: {ee_err}")
        st.stop()
    
    st.sidebar.success("✅ Earth Engine Connected")
    
    # Sidebar parameters
    with st.sidebar:
        st.header("📅 Date Range")
        start_date = st.date_input(
            "Start Date",
            value=datetime.date(2025, 1, 1),
            min_value=datetime.date(2012, 1, 1),
            max_value=datetime.date.today()
        )
        end_date = st.date_input(
            "End Date",
            value=datetime.date(2025, 2, 1),
            min_value=datetime.date(2012, 1, 1),
            max_value=datetime.date.today()
        )
        
        st.divider()
        
        # Scale parameters
        st.subheader("🔬 Resolution Settings")
        ndvi_scale = st.slider(
            "NDVI Scale (degrees)",
            min_value=0.001,
            max_value=0.01,
            value=0.001,
            step=0.001,
            format="%.3f"
        )
        lst_scale = st.slider(
            "LST Scale (degrees)",
            min_value=0.001,
            max_value=0.01,
            value=0.005,
            step=0.001,
            format="%.3f"
        )
        
        st.divider()
        
        load_button = st.button("🚀 Load & Process Data", type="primary", use_container_width=True)
        
        st.divider()
        st.markdown("""
        **📊 Data Sources:**
        - **NDVI:** NASA/VIIRS/002/VNP09GA
        - **LST:** NASA/VIIRS/002/VNP21A1D
        """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Draw & Select Area", "🌿 NDVI Analysis", "🌡️ LST Analysis"])
    
    with tab1:
        st.subheader("Draw Your Area of Interest")
        st.info("🔽 Use the rectangle tool (□) in the top-left corner of the map to draw your area")
        
        # Create and display map
        m = create_map()
        
        # Store drawn features in session state
        if 'last_draw' not in st.session_state:
            st.session_state.last_draw = None
        
        # Display the map and capture drawing
        output = st_folium(
            m,
            width=900,
            height=500,
            key="draw_map"
        )
        
        # Check if a rectangle was drawn
        if output and 'last_active_drawing' in output:
            last_draw = output['last_active_drawing']
            if last_draw:
                st.session_state.last_draw = last_draw
                st.success("✅ Rectangle drawn! Click 'Load & Process Data' in the sidebar to analyze.")
                
                # Display the coordinates
                if 'geometry' in last_draw:
                    coords = last_draw['geometry']['coordinates'][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    
                    st.write("**Your Area of Interest:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Longitude: {min(lons):.4f} to {max(lons):.4f}")
                    with col2:
                        st.write(f"Latitude: {min(lats):.4f} to {max(lats):.4f}")
        
        elif st.session_state.last_draw is None:
            st.info("ℹ️ No rectangle drawn yet. Draw one on the map to begin.")
        else:
            st.info("ℹ️ Use the rectangle tool to draw your area of interest")
    
    # Process data when button is clicked
    if load_button:
        if st.session_state.last_draw is None:
            st.error("❌ Please draw a rectangle on the map first!")
        else:
            with st.spinner("Loading data from Earth Engine..."):
                # Get ROI from drawing
                geojson = st.session_state.last_draw['geometry']
                
                # Convert GeoJSON to ee.Geometry
                if geojson['type'] == 'Polygon':
                    coords = geojson['coordinates'][0]
                    # Create ee.Geometry from coordinates
                    roi = ee.Geometry.Polygon(coords)
                    
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    
                    # Status bar
                    progress_text = st.empty()
                    
                    # Load NDVI
                    progress_text.text("Loading NDVI data...")
                    ndvi_ds, ndvi_count = get_ndvi_data(roi, start_str, end_str, ndvi_scale)
                    
                    # Load LST
                    progress_text.text("Loading LST data...")
                    lst_ds, lst_count = get_lst_data(roi, start_str, end_str, lst_scale)
                    
                    progress_text.text("Processing complete!")
                    progress_text.empty()
                    
                    # Store in session state
                    st.session_state.ndvi_ds = ndvi_ds
                    st.session_state.ndvi_count = ndvi_count
                    st.session_state.lst_ds = lst_ds
                    st.session_state.lst_count = lst_count
                    st.session_state.roi = roi
                    
                    st.success(f"✅ Data loaded! Found {ndvi_count} NDVI images and {lst_count} LST images")
                else:
                    st.error("Please draw a rectangle (not another shape)")
    
    # Display NDVI analysis
    with tab2:
        st.subheader("🌿 NDVI Analysis")
        
        if 'ndvi_ds' in st.session_state and st.session_state.ndvi_ds is not None:
            ds = st.session_state.ndvi_ds
            count = st.session_state.ndvi_count
            
            st.metric("Number of NDVI Images", count)
            
            # Show data shape
            st.write(f"**Data Dimensions:** Time: {len(ds.time)} | Lat: {len(ds.lat)} | Lon: {len(ds.lon)}")
            
            # Plot time series grid
            st.write("### NDVI Time Series Plots")
            fig = plot_ndvi_grid(ds)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            # Show mean NDVI over time
            st.write("### Mean NDVI Over Time")
            mean_ndvi = ds.ndvi.mean(dim=['lat', 'lon'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            dates = [str(t)[:10] for t in ds.time.values]
            ax.plot(dates, mean_ndvi.values, 'o-', color='green', linewidth=2, markersize=6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Mean NDVI')
            ax.set_title('NDVI Time Series - Area Average')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Raw data table
            with st.expander("View Raw Data"):
                df = mean_ndvi.to_dataframe().reset_index()
                df['time'] = df['time'].astype(str).str[:10]
                st.dataframe(df, use_container_width=True)
                
        else:
            st.info("No NDVI data loaded yet. Draw a rectangle and click 'Load & Process Data'.")
    
    # Display LST analysis
    with tab3:
        st.subheader("🌡️ LST Analysis")
        
        if 'lst_ds' in st.session_state and st.session_state.lst_ds is not None:
            ds = st.session_state.lst_ds
            count = st.session_state.lst_count
            
            st.metric("Number of LST Images", count)
            
            # Show data shape
            st.write(f"**Data Dimensions:** Time: {len(ds.time)} | Lat: {len(ds.lat)} | Lon: {len(ds.lon)}")
            
            # Plot time series grid
            st.write("### LST Time Series Plots")
            fig = plot_lst_grid(ds)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            # Show mean LST over time (convert to Celsius)
            st.write("### Mean LST Over Time (Celsius)")
            mean_lst = ds.LST_1KM.mean(dim=['lat', 'lon'])
            mean_lst_celsius = mean_lst - 273.15
            
            fig, ax = plt.subplots(figsize=(10, 4))
            dates = [str(t)[:10] for t in ds.time.values]
            ax.plot(dates, mean_lst_celsius.values, 'o-', color='red', linewidth=2, markersize=6)
            ax.set_xlabel('Date')
            ax.set_ylabel('Mean LST (°C)')
            ax.set_title('LST Time Series - Area Average')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Raw data table
            with st.expander("View Raw Data"):
                df = mean_lst_celsius.to_dataframe().reset_index()
                df['time'] = df['time'].astype(str).str[:10]
                df['LST_1KM'] = df['LST_1KM'] - 273.15  # Convert to Celsius
                df.columns = ['time', 'Mean LST (°C)']
                st.dataframe(df, use_container_width=True)
                
        else:
            st.info("No LST data loaded yet. Draw a rectangle and click 'Load & Process Data'.")


if __name__ == "__main__":
    main()
