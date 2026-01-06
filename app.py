import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Employee Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #e9ecef;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df = df[df['total_actions'] > 0].copy()
        
        # Convert DOJ and calculate tenure
        df['DOJ'] = pd.to_datetime(df['DOJ'], errors='coerce')
        df['tenure_months'] = ((datetime.now() - df['DOJ']).dt.days / 30).round(1)
        
        # Calculate performance metrics
        df['efficiency_score'] = (df['total_actions'] / (df['avg_engage_per_ticket_secs'] / 60)).round(2)
        df['consistency_score'] = (100 / (1 + df['engage_std_dev_seconds'])).round(2)
        
        # Calculate percentages for time buckets
        buckets = [
            'engage_under_15s', 'engage_15s_to_20s', 'engage_20s_to_25s',
            'engage_25s_to_30s', 'engage_30s_to_35s', 'engage_35s_to_40s',
            'engage_40s_to_45s', 'engage_45s_to_50s', 'engage_50s_to_55s',
            'engage_55s_to_60s', 'engage_over_60s'
        ]
        
        for bucket in buckets:
            if bucket in df.columns:
                df[f'{bucket}_pct'] = ((df[bucket] / df['total_actions']) * 100).round(1)
        
        # Calculate medians
        median_time = df['avg_engage_per_ticket_secs'].median()
        median_actions = df['total_actions'].median()
        
        # Assign quadrants
        def assign_quadrant(row):
            if row['avg_engage_per_ticket_secs'] <= median_time and row['total_actions'] >= median_actions:
                return '🏆 Stars'
            elif row['avg_engage_per_ticket_secs'] <= median_time and row['total_actions'] < median_actions:
                return '⚡ Speed Demons'
            elif row['avg_engage_per_ticket_secs'] > median_time and row['total_actions'] >= median_actions:
                return '🐢 Workhorses'
            else:
                return '🎯 Needs Support'
        
        df['quadrant'] = df.apply(assign_quadrant, axis=1)
        df['quadrant_detailed'] = df['quadrant'].map({
            '🏆 Stars': '🏆 Stars (Fast + High Volume)',
            '⚡ Speed Demons': '⚡ Speed Demons (Fast + Low Volume)',
            '🐢 Workhorses': '🐢 Workhorses (Slow + High Volume)',
            '🎯 Needs Support': '🎯 Needs Support (Slow + Low Volume)'
        })
        
        return df, median_time, median_actions
    
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        return None, None, None

# Create chart function
def create_chart(filtered_df, full_df, median_time, median_actions):
    quadrant_colors = {
        '🏆 Stars': '#27ae60',
        '⚡ Speed Demons': '#f39c12',
        '🐢 Workhorses': '#3498db',
        '🎯 Needs Support': '#e74c3c'
    }

    fig = go.Figure()

    # Add scatter traces for each quadrant
    for quadrant in ['🏆 Stars', '⚡ Speed Demons', '🐢 Workhorses', '🎯 Needs Support']:
        df_quad = filtered_df[filtered_df['quadrant'] == quadrant]

        if len(df_quad) == 0:
            continue

        # Create detailed hover text
        hover_text = []
        for idx, row in df_quad.iterrows():
            # Check if percentage columns exist
            quick_pct = row.get('engage_under_15s_pct', 0)
            normal_pct = (row.get('engage_15s_to_20s_pct', 0) + 
                         row.get('engage_20s_to_25s_pct', 0) + 
                         row.get('engage_25s_to_30s_pct', 0))
            extended_pct = (row.get('engage_30s_to_35s_pct', 0) + 
                           row.get('engage_35s_to_40s_pct', 0) + 
                           row.get('engage_40s_to_45s_pct', 0) + 
                           row.get('engage_45s_to_50s_pct', 0) + 
                           row.get('engage_50s_to_55s_pct', 0) + 
                           row.get('engage_55s_to_60s_pct', 0))
            long_pct = row.get('engage_over_60s_pct', 0)

            text = (
                f"<b style='font-size:14px'>{row['employee_name']}</b><br>"
                f"<b>━━━━━━━━━━━━━━━━━━━━━━━━━━</b><br>"
                f"<br><b>📊 Core Metrics:</b><br>"
                f"  • Avg Engagement: <b>{row['avg_engage_per_ticket_secs']:.1f}s</b><br>"
                f"  • Total Actions: <b>{row['total_actions']:.0f}</b><br>"
                f"  • Efficiency: <b>{row['efficiency_score']:.2f}</b> actions/min<br>"
                f"  • Consistency: <b>{row['consistency_score']:.2f}</b><br>"
                f"  • Std Deviation: {row['engage_std_dev_seconds']:.1f}s<br>"
                f"<br><b>⏱️ Time Distribution:</b><br>"
                f"  • Quick (<15s): {quick_pct:.1f}%<br>"
                f"  • Normal (15-30s): {normal_pct:.1f}%<br>"
                f"  • Extended (30-60s): {extended_pct:.1f}%<br>"
                f"  • Long (>60s): {long_pct:.1f}%<br>"
                f"<br><b>👤 Employee Info:</b><br>"
                f"  • Experience: {row['tenure_months']:.1f} months<br>"
                f"  • Category: <b>{row['quadrant_detailed']}</b>"
            )
            hover_text.append(text)

        fig.add_trace(go.Scatter(
            x=df_quad['avg_engage_per_ticket_secs'],
            y=df_quad['total_actions'],
            mode='markers',
            name=df_quad['quadrant_detailed'].iloc[0],
            marker=dict(
                size=np.sqrt(df_quad['efficiency_score']) * 8,
                color=quadrant_colors[quadrant],
                line=dict(width=2, color='white'),
                opacity=0.85,
                symbol='circle'
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
        ))

    # Get axis ranges from full dataset
    x_min, x_max = full_df['avg_engage_per_ticket_secs'].min() * 0.95, full_df['avg_engage_per_ticket_secs'].max() * 1.05
    y_min, y_max = full_df['total_actions'].min() * 0.95, full_df['total_actions'].max() * 1.05

    # Add median lines
    fig.add_hline(
        y=median_actions, 
        line_dash="dash", 
        line_color="rgba(231, 76, 60, 0.7)", 
        line_width=3,
        annotation=dict(
            text=f"<b>Median Actions: {median_actions:.0f}</b>",
            font=dict(size=13, color="white", family="Arial Black"),
            bgcolor="rgba(231, 76, 60, 0.8)",
            borderpad=6
        ),
        annotation_position="right"
    )

    fig.add_vline(
        x=median_time, 
        line_dash="dash", 
        line_color="rgba(52, 152, 219, 0.7)", 
        line_width=3,
        annotation=dict(
            text=f"<b>Median Time: {median_time:.1f}s</b>",
            font=dict(size=13, color="white", family="Arial Black"),
            bgcolor="rgba(52, 152, 219, 0.8)",
            borderpad=6
        ),
        annotation_position="top"
    )

    # Add quadrant background shading
    fig.add_shape(type="rect", x0=x_min, y0=median_actions, x1=median_time, y1=y_max,
                  fillcolor="rgba(39, 174, 96, 0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=median_time, y0=median_actions, x1=x_max, y1=y_max,
                  fillcolor="rgba(52, 152, 219, 0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=median_time, y1=median_actions,
                  fillcolor="rgba(243, 156, 18, 0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=median_time, y0=y_min, x1=x_max, y1=median_actions,
                  fillcolor="rgba(231, 76, 60, 0.08)", line_width=0, layer="below")

    # Add quadrant labels
    fig.add_annotation(
        x=median_time * 0.5, y=y_max * 0.92,
        text="<b>🏆 STARS</b><br><i>Fast & High Volume</i><br>Keep It Up! 🎉",
        showarrow=False,
        font=dict(size=15, color='#27ae60', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#27ae60',
        borderwidth=2,
        borderpad=10
    )

    fig.add_annotation(
        x=x_max * 0.92, y=y_max * 0.92,
        text="<b>🐢 WORKHORSES</b><br><i>Thorough & High Volume</i><br>Speed Training 🚀",
        showarrow=False,
        font=dict(size=15, color='#3498db', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#3498db',
        borderwidth=2,
        borderpad=10
    )

    fig.add_annotation(
        x=median_time * 0.5, y=y_min * 1.25,
        text="<b>⚡ SPEED DEMONS</b><br><i>Fast but Low Volume</i><br>Boost Output 📈",
        showarrow=False,
        font=dict(size=15, color='#f39c12', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#f39c12',
        borderwidth=2,
        borderpad=10
    )

    fig.add_annotation(
        x=x_max * 0.92, y=y_min * 1.25,
        text="<b>🎯 NEEDS SUPPORT</b><br><i>Development Zone</i><br>Training Focus 🎓",
        showarrow=False,
        font=dict(size=15, color='#e74c3c', family='Arial Black'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#e74c3c',
        borderwidth=2,
        borderpad=10
    )

    # Update layout
    fig.update_layout(
        title={
            'text': (
                "⏱️ <b>Engagement Time vs Total Actions</b> 📊<br>"
                "<sup style='font-size:14px'>Interactive Performance Quadrant Analysis | "
                "Hover for details | Zoom & Pan enabled</sup>"
            ),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        xaxis=dict(
            title=dict(
                text="<b>⏱️ Average Engagement Time per Ticket (seconds)</b>",
                font=dict(size=14, color='#34495e', family='Arial')
            ),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True,
            zeroline=False,
            range=[x_min, x_max]
        ),
        yaxis=dict(
            title=dict(
                text="<b>📊 Total Actions Completed</b>",
                font=dict(size=14, color='#34495e', family='Arial')
            ),
            gridcolor='rgba(189, 195, 199, 0.3)',
            showgrid=True,
            zeroline=False,
            range=[y_min, y_max]
        ),
        height=700,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#f8f9fa',
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="#2c3e50"
        ),
        legend=dict(
            title=dict(text="<b>Performance Categories</b>", font=dict(size=12)),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#34495e",
            borderwidth=2,
            font=dict(size=11, family='Arial'),
        ),
        font=dict(family='Arial', size=11, color='#2c3e50')
    )

    return fig

def main():
    st.markdown('<div class="main-header">⏱️ Employee Performance Dashboard 📊</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive Quadrant Analysis with Advanced Filters</div>', unsafe_allow_html=True)
    
    # FILE UPLOADER - ADD THIS
    st.markdown("### 📤 Upload Your Data File")
    uploaded_file = st.file_uploader(
        "Choose your employee data CSV file",
        type=['csv'],
        help="Upload the main_sheet.csv.csv file with employee performance data"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload your CSV file to get started")
        st.markdown("#### Required Columns:")
        st.markdown("""
        - `employee_name`, `employee_id`
        - `DOJ` (Date of Joining)
        - `total_actions`, `avg_engage_per_ticket_secs`
        - `engage_std_dev_seconds`
        - Time buckets: `engage_under_15s`, `engage_15s_to_20s`, etc.
        """)
        st.stop()
    
    # Load data from uploaded file
    df, median_time, median_actions = load_data(uploaded_file)
    
    if df is None:
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(df)} employees!")
    st.markdown("---")
    
    # REST OF YOUR CODE CONTINUES HERE...
    # (Keep all the sidebar filters and chart code as is)


    # Sidebar filters
    st.sidebar.header("🔍 Filter Options")
    st.sidebar.markdown("---")

    # 1. Quadrant Filter
    st.sidebar.subheader("📊 Performance Category")
    selected_quadrants = st.sidebar.multiselect(
        "Select Quadrants",
        options=['🏆 Stars', '⚡ Speed Demons', '🐢 Workhorses', '🎯 Needs Support'],
        default=['🏆 Stars', '⚡ Speed Demons', '🐢 Workhorses', '🎯 Needs Support'],
        help="Filter by performance quadrant"
    )

    # 2. Tenure Filter
    st.sidebar.subheader("📅 Employee Tenure")
    tenure_range = st.sidebar.slider(
        "Tenure Range (months)",
        min_value=float(df['tenure_months'].min()),
        max_value=float(df['tenure_months'].max()),
        value=(float(df['tenure_months'].min()), float(df['tenure_months'].max())),
        help="Filter by employee experience"
    )

    # 3. Efficiency Score Filter
    st.sidebar.subheader("⚡ Efficiency Score")
    efficiency_range = st.sidebar.slider(
        "Efficiency Range (actions/min)",
        min_value=float(df['efficiency_score'].min()),
        max_value=float(df['efficiency_score'].max()),
        value=(float(df['efficiency_score'].min()), float(df['efficiency_score'].max())),
        help="Filter by efficiency score"
    )

    # 4. Total Actions Filter
    st.sidebar.subheader("📈 Total Actions")
    actions_range = st.sidebar.slider(
        "Actions Range",
        min_value=int(df['total_actions'].min()),
        max_value=int(df['total_actions'].max()),
        value=(int(df['total_actions'].min()), int(df['total_actions'].max())),
        help="Filter by total actions completed"
    )

    # 5. Average Time Filter
    st.sidebar.subheader("⏱️ Avg Engagement Time")
    time_range = st.sidebar.slider(
        "Time Range (seconds)",
        min_value=float(df['avg_engage_per_ticket_secs'].min()),
        max_value=float(df['avg_engage_per_ticket_secs'].max()),
        value=(float(df['avg_engage_per_ticket_secs'].min()), 
               float(df['avg_engage_per_ticket_secs'].max())),
        help="Filter by average engagement time"
    )

    # 6. Employee Search
    st.sidebar.subheader("🔎 Search Employee")
    search_query = st.sidebar.text_input(
        "Employee Name",
        placeholder="Type to search...",
        help="Search by employee name"
    )

    # Reset button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters"):
        st.rerun()

    # Apply filters
    filtered_df = df[
        (df['quadrant'].isin(selected_quadrants)) &
        (df['tenure_months'].between(tenure_range[0], tenure_range[1])) &
        (df['efficiency_score'].between(efficiency_range[0], efficiency_range[1])) &
        (df['total_actions'].between(actions_range[0], actions_range[1])) &
        (df['avg_engage_per_ticket_secs'].between(time_range[0], time_range[1]))
    ]

    if search_query:
        filtered_df = filtered_df[filtered_df['employee_name'].str.contains(search_query, case=False, na=False)]

    # Show metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Filtered Employees", len(filtered_df), f"{len(filtered_df)/len(df)*100:.1f}%")

    with col2:
        st.metric("⚡ Avg Efficiency", f"{filtered_df['efficiency_score'].mean():.2f}", "actions/min")

    with col3:
        st.metric("⏱️ Avg Time", f"{filtered_df['avg_engage_per_ticket_secs'].mean():.1f}s")

    with col4:
        st.metric("📊 Avg Actions", f"{filtered_df['total_actions'].mean():.0f}")

    st.markdown("---")

    # Show chart
    if len(filtered_df) > 0:
        fig = create_chart(filtered_df, df, median_time, median_actions)
        st.plotly_chart(fig, use_container_width=True)

        # Quadrant distribution
        st.markdown("### 📊 Quadrant Distribution")
        quad_cols = st.columns(4)

        for idx, quadrant in enumerate(['🏆 Stars', '⚡ Speed Demons', '🐢 Workhorses', '🎯 Needs Support']):
            count = len(filtered_df[filtered_df['quadrant'] == quadrant])
            pct = (count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            with quad_cols[idx]:
                st.metric(quadrant, f"{count}", f"{pct:.1f}%")

        st.markdown("---")

        # Data table
        st.markdown("### 📋 Filtered Employee Data")

        display_df = filtered_df[['employee_name', 'quadrant', 'avg_engage_per_ticket_secs', 
                                   'total_actions', 'efficiency_score', 'consistency_score', 
                                   'tenure_months']].sort_values(by='efficiency_score', ascending=False)

        display_df.columns = ['Employee', 'Category', 'Avg Time (s)', 'Total Actions', 
                              'Efficiency', 'Consistency', 'Tenure (months)']

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f'employee_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
            use_container_width=True
        )

    else:
        st.warning("⚠️ No employees match the selected filters. Please adjust your filter criteria.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #7f8c8d;'>"
        "📊 Employee Performance Dashboard | Built with Streamlit & Plotly | "
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
