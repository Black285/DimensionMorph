from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import base64
import os
import json
from typing import Any
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# Custom JSON encoder to handle NumPy types and non-finite floats
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isinf(obj) or np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_eval(expression, x=None, y=None, parameters=None) -> Any:
    """Safely evaluate mathematical expression with x, y variables and optional parameters"""
    def safe_prod(values):
        result = None
        for value in values:
            if result is None:
                result = value
            else:
                result = result * value
        return 1 if result is None else result

    def safe_sub(*values):
        if len(values) == 0:
            return 0
        result = values[0]
        for value in values[1:]:
            result = result - value
        return result

    def safe_div(*values):
        if len(values) == 0:
            return 1
        result = values[0]
        for value in values[1:]:
            result = result / value
        return result

    allowed = {
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsin': np.arcsin,
        'asin': np.arcsin,
        'arccos': np.arccos,
        'acos': np.arccos,
        'arctan': np.arctan,
        'atan': np.arctan,
        'atan2': np.arctan2,
        'arcsinh': np.arcsinh,
        'asinh': np.arcsinh,
        'arccosh': np.arccosh,
        'acosh': np.arccosh,
        'arctanh': np.arctanh,
        'atanh': np.arctanh,
        # Exponential & Logarithmic
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'log2': np.log2,
        'sqrt': np.sqrt,
        'power': np.power,
        'pow': np.power,
        # Rounding & Absolute value
        'abs': np.abs,
        'ceil': np.ceil,
        'floor': np.floor,
        'round': np.round,
        'sign': np.sign,
        # Min/Max
        'min': np.minimum,
        'max': np.maximum,
        # Series & aggregation
        'sum': np.sum,
        'dif': np.diff,
        'diff': np.diff,
        'sub': safe_sub,
        'prod': safe_prod,
        'mul': safe_prod,
        'div': safe_div,
        'range': range,
        # Constants
        'pi': np.pi,
        'e': np.e,
        'x': x,
        'y': y,
    }
    if parameters:
        allowed.update(parameters)

    safe_builtins = {'__builtins__': {}}
    try:
        return eval(expression, safe_builtins, allowed)
    except Exception:
        raise ValueError(f"Invalid expression: {expression}")


def interpolate_invalid_1d(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values

    valid = np.isfinite(values)
    if not np.any(valid):
        return np.array([], dtype=np.float64)
    if np.all(valid):
        return values

    indices = np.arange(values.size)
    values = values.copy()
    values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
    return values


def get_dynamic_limits(x_values, y_values, square_aspect=False):
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    if x_values.size == 0 or y_values.size == 0:
        return (-1.0, 1.0)

    limit = max(float(np.max(np.abs(x_values))), float(np.max(np.abs(y_values)))) * 1.1
    if square_aspect:
        return (-limit, limit)
    return (float(np.min(x_values)), float(np.max(x_values)))


def plot_function_transformation(expression, x_range=(-3, 3), num_points=500, parameters=None):
    try:
        x = np.linspace(x_range[0], x_range[1], num_points)
        x = x[np.abs(x) > 0.05]

        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            y = safe_eval(expression, x=x, parameters=parameters)

        if isinstance(y, complex):
            y = np.real(y)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.float64)
        else:
            y = y.astype(np.float64, copy=False)

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        if len(x_valid) == 0:
            return None, 'Function produced no valid values'

        r = np.sqrt(x_valid ** 2 + y_valid ** 2)
        theta = np.arctan2(y_valid, x_valid)

        theta_domain = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            r_polar = safe_eval(expression, x=theta_domain, parameters=parameters)

        if isinstance(r_polar, complex):
            r_polar = np.real(r_polar)
        if not isinstance(r_polar, np.ndarray):
            r_polar = np.asarray(r_polar, dtype=np.float64)
        else:
            r_polar = r_polar.astype(np.float64, copy=False)
        r_polar = interpolate_invalid_1d(np.abs(r_polar))

        plot1 = go.Figure()
        plot1.add_trace(go.Scatter(
            x=x_valid.tolist(),
            y=y_valid.tolist(),
            mode='lines+markers',
            name=f'y = {expression}',
            line=dict(color='#667eea', width=2),
            marker=dict(size=3, opacity=0.3)
        ))
        limit = max(float(np.max(np.abs(x_valid))), float(np.max(np.abs(y_valid)))) * 1.1
        plot1.update_layout(
            title='1. Cartesian (Original)',
            xaxis=dict(title='x', range=[-limit, limit]),
            yaxis=dict(title='y', range=[-limit, limit]),
            autosize=False, width=600, height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # ===== PLOT 2: Polar (θ-r Cartesian space) =====
        theta_unwrapped = np.unwrap(theta)
        r_limit = float(np.max(r)) * 1.1 if len(r) > 0 else 1.0
        plot2 = go.Figure()
        plot2.add_trace(go.Scatter(
            x=theta_unwrapped.tolist(),
            y=r.tolist(),
            mode='lines+markers',
            name='Polar (θ-r)',
            line=dict(color='#ff6b6b', width=2),
            marker=dict(size=3, opacity=0.3)
        ))
        plot2.add_hline(y=0, line=dict(color='black', width=1.5), opacity=0.7)
        plot2.add_vline(x=0, line=dict(color='black', width=1.5), opacity=0.7)
        plot2.update_layout(
            title='2. Polar (θ-r Cartesian)',
            xaxis=dict(title='θ (radians)', range=[-np.pi, np.pi]),
            yaxis=dict(title='r (radius)', range=[0, r_limit]),
            autosize=False, width=600, height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        theta_degrees = np.degrees(theta_domain)
        plot3 = go.Figure()
        plot3.add_trace(go.Scatterpolar(
            r=r_polar.tolist(),
            theta=theta_degrees.tolist(),
            mode='lines+markers',
            name='r = |f(θ)|',
            line=dict(color='#51cf66', width=2),
            marker=dict(size=3, opacity=0.3)
        ))
        plot3.update_layout(
            title='3. Polar (Function as Radius)',
            polar=dict(
                radialaxis=dict(visible=True),
                angularaxis=dict(direction='counterclockwise')
            ),
            autosize=False, width=600, height=600,
            margin=dict(l=50, r=50, t=60, b=50),
            paper_bgcolor='white'
        )

        # Convert each figure to a PNG image via Plotly/kaleido
        images = []
        for fig in [plot1, plot2, plot3]:
            img_bytes = pio.to_image(fig, format='png', width=600, height=600, scale=1.5)
            images.append(base64.b64encode(img_bytes).decode('utf-8'))

        return images, None

    except Exception as e:
        return None, str(e)

def plot_3d_matplotlib(expression, x_range=(-3, 3), y_range=(-3, 3), num_points=30, parameters=None):
    """Generate 3D plots as PNG images by reusing the Plotly 3D pipeline and converting via kaleido."""
    try:
        fig_jsons, error = plot_3d_function_transformation(
            expression, x_range, y_range, num_points, parameters
        )
        if error:
            return None, error

        images = []
        for fig_json in fig_jsons:
            fig_dict = json.loads(fig_json)
            fig = go.Figure(fig_dict)
            img_bytes = pio.to_image(fig, format='png', width=700, height=600, scale=1.5)
            images.append(base64.b64encode(img_bytes).decode('utf-8'))

        return images, None

    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/icon.png')
def serve_icon():
    icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    return send_file(icon_path, mimetype='image/png')

def plot_3d_function_transformation(expression, x_range=(-3, 3), y_range=(-3, 3), num_points=30, parameters=None):
    """Generate Plotly 3D plots for 3D function in Cartesian and Spherical coordinates"""
    try:
        # Generate x, y mesh
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        
        # Filter out values too close to zero to avoid singularities
        x = x[np.abs(x) > 0.05]
        y = y[np.abs(y) > 0.05]
        
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function with both x and y
        Z = safe_eval(expression, x=X, y=Y, parameters=parameters)
        if Z is None:
            raise ValueError("Function evaluation failed")
        if isinstance(Z, (int, float)):
            Z = np.full_like(X, Z, dtype=float)
        
        # Handle complex numbers
        if np.iscomplexobj(Z):
            Z = np.real(Z)
        
        # Filter out NaN and inf values
        valid_mask = np.isfinite(Z)
        if not np.any(valid_mask):
            raise ValueError("Function produced no valid values in the given range")
        
        # Transform to spherical coordinates
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arccos(Z / np.clip(r, 1e-10, None))
        
        # Unwrap theta to avoid discontinuity artifacts when plotting as a surface
        # Unwrap along each row to handle the mesh structure properly
        theta_unwrapped = np.copy(theta)
        for i in range(theta_unwrapped.shape[0]):
            theta_unwrapped[i, :] = np.unwrap(theta_unwrapped[i, :])
        
        # Calculate data ranges with padding to prevent clipping
        r_min, r_max = np.nanmin(r), np.nanmax(r)
        r_padding = (r_max - r_min) * 0.1 if r_max > r_min else 0.5
        
        theta_min, theta_max = np.nanmin(theta_unwrapped), np.nanmax(theta_unwrapped)
        theta_padding = (theta_max - theta_min) * 0.1 if theta_max > theta_min else 0.5
        
        phi_min, phi_max = np.nanmin(phi), np.nanmax(phi)
        phi_padding = (phi_max - phi_min) * 0.1 if phi_max > phi_min else 0.5
        
        # CRITICAL: Convert numpy arrays to Python lists for JSON serialization
        # Replace any inf or nan values with None to avoid JSON serialization errors
        def clean_array_for_json(arr):
            """Convert numpy array to list, replacing inf/nan with None"""
            arr = np.asarray(arr, dtype=float)
            arr = arr.copy()
            # Replace infinite and NaN values with a safe placeholder
            mask = ~np.isfinite(arr)
            arr[mask] = 0  # Replace with 0 instead of NaN to keep valid JSON
            return arr.tolist()
        
        X_list = clean_array_for_json(X)
        Y_list = clean_array_for_json(Y)
        Z_list = clean_array_for_json(Z)
        r_list = clean_array_for_json(r)
        theta_unwrapped_list = clean_array_for_json(theta_unwrapped)
        phi_list = clean_array_for_json(phi)
        
        # ===== PLOT 1: Original Cartesian 3D (Surface) =====
        fig1 = go.Figure(data=[go.Surface(x=X_list, y=Y_list, z=Z_list, colorscale='viridis', name='Surface')])
        fig1.update_layout(
            title='1. Original (Cartesian X-Y-Z)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # ===== PLOT 2: Spherical as Cartesian axes (r, theta, phi) =====
        # Use unwrapped theta to avoid discontinuity jumps in the surface
        # Set explicit axis ranges to prevent clipping of data
        fig2 = go.Figure(data=[go.Surface(x=r_list, y=theta_unwrapped_list, z=phi_list, colorscale='plasma', name='Surface')])
        fig2.update_layout(
            title='2. Spherical (r-θ-φ Cartesian)',
            scene=dict(
                xaxis=dict(
                    title='r (radius)',
                    range=[r_min - r_padding, r_max + r_padding]
                ),
                yaxis=dict(
                    title='θ (azimuthal, unwrapped)',
                    range=[theta_min - theta_padding, theta_max + theta_padding]
                ),
                zaxis=dict(
                    title='φ (polar)',
                    range=[phi_min - phi_padding, phi_max + phi_padding]
                )
            ),
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # ===== PLOT 3: 3D Spherical Surface =====
        # Create a 3D surface in spherical coordinates where radius varies with both theta and phi
        
        fig3 = go.Figure()
        
        # Create a grid of theta and phi values with wraparound to avoid θ discontinuity
        theta_grid = np.linspace(-np.pi, np.pi, 60)
        # Add a wraparound column at 2π (equivalent to -π) to create a seamless surface
        theta_grid_wrapped = np.append(theta_grid, 2*np.pi)
        phi_grid = np.linspace(0.1, np.pi - 0.1, 40)  # Avoid singularities at 0 and π
        THETA, PHI = np.meshgrid(theta_grid_wrapped, phi_grid)
        
        # Evaluate function at each (theta, phi) point
        R = safe_eval(expression, x=THETA, y=PHI, parameters=parameters)
        if isinstance(R, (int, float)):
            R = np.full_like(THETA, R)
        R = np.abs(np.asarray(R, dtype=float))
        
        # Filter out infinite and NaN values to avoid JSON serialization errors
        finite_mask = np.isfinite(R)
        if not np.any(finite_mask):
            raise ValueError("Function produced no valid values")
        R[~finite_mask] = 0  # Replace inf/nan with 0
        
        # Normalize radius for visualization (prevent too large/small values)
        r_max = np.max(np.abs(R)) + 1e-6
        if not np.isfinite(r_max) or r_max == 0:
            r_max = 1.0
        r_norm = np.abs(R) / r_max
        
        # Ensure r_norm doesn't contain inf/nan
        r_norm[~np.isfinite(r_norm)] = 0
        
        # Convert spherical coordinates to Cartesian
        X_sphere = r_norm * np.sin(PHI) * np.cos(THETA)
        Y_sphere = r_norm * np.sin(PHI) * np.sin(THETA)
        Z_sphere = r_norm * np.cos(PHI)
        
        # Clean arrays before JSON serialization
        X_sphere = clean_array_for_json(X_sphere) 
        Y_sphere = clean_array_for_json(Y_sphere)
        Z_sphere = clean_array_for_json(Z_sphere)
        
        # Create the 3D surface (now seamless due to wraparound)
        fig3.add_trace(go.Surface(
            x=X_sphere if isinstance(X_sphere, list) else X_sphere.tolist(),
            y=Y_sphere if isinstance(Y_sphere, list) else Y_sphere.tolist(),
            z=Z_sphere if isinstance(Z_sphere, list) else Z_sphere.tolist(),
            colorscale='Viridis',
            colorbar=dict(title='r'),
            name='Spherical Surface',
            hoverinfo='skip'
        ))
        
        # Add reference grid circles for context (theta plane)
        num_grid_circles = 5
        for grid_idx in range(1, num_grid_circles + 1):
            radius_frac = grid_idx / num_grid_circles
            theta_ref = np.linspace(0, 2*np.pi, 100)
            x_ref = radius_frac * np.cos(theta_ref)
            y_ref = radius_frac * np.sin(theta_ref)
            z_ref = np.zeros_like(theta_ref)
            
            fig3.add_trace(go.Scatter3d(
                x=x_ref.tolist(),
                y=y_ref.tolist(),
                z=z_ref.tolist(),
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add reference radial lines (theta plane)
        num_radii_theta = 12
        for i in range(num_radii_theta):
            angle = (i / num_radii_theta) * 2 * np.pi
            x_rad = [0, np.cos(angle)]
            y_rad = [0, np.sin(angle)]
            z_rad = [0, 0]
            
            fig3.add_trace(go.Scatter3d(
                x=x_rad,
                y=y_rad,
                z=z_rad,
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add angle label on theta plane
            angle_deg = round(np.degrees(angle))
            label_dist = 1.15
            fig3.add_trace(go.Scatter3d(
                x=[label_dist * np.cos(angle)],
                y=[label_dist * np.sin(angle)],
                z=[0],
                mode='text',
                text=[f'{angle_deg}°'],
                textposition='middle center',
                textfont=dict(size=8, color='blue'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add radius labels on theta plane
        for grid_idx in range(1, num_grid_circles + 1):
            radius_frac = grid_idx / num_grid_circles
            fig3.add_trace(go.Scatter3d(
                x=[radius_frac + 0.05],
                y=[0],
                z=[0],
                mode='text',
                text=[f'r={radius_frac:.1f}'],
                textposition='middle center',
                textfont=dict(size=8, color='blue'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add reference grid circles for phi plane
        for grid_idx in range(1, num_grid_circles + 1):
            radius_frac = grid_idx / num_grid_circles
            phi_ref = np.linspace(0, 2*np.pi, 100)
            x_ref = radius_frac * np.cos(phi_ref)
            y_ref = np.zeros_like(phi_ref)
            z_ref = radius_frac * np.sin(phi_ref)
            
            fig3.add_trace(go.Scatter3d(
                x=x_ref.tolist(),
                y=y_ref.tolist(),
                z=z_ref.tolist(),
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add reference radial lines (phi plane)
        num_radii_phi = 12
        for i in range(num_radii_phi):
            angle = (i / num_radii_phi) * 2 * np.pi
            x_rad = [0, np.cos(angle)]
            y_rad = [0, 0]
            z_rad = [0, np.sin(angle)]
            
            fig3.add_trace(go.Scatter3d(
                x=x_rad,
                y=y_rad,
                z=z_rad,
                mode='lines',
                line=dict(color='rgba(0, 0, 0, 0.3)', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add angle label on phi plane
            angle_deg = round(np.degrees(angle))
            label_dist = 1.15
            fig3.add_trace(go.Scatter3d(
                x=[label_dist * np.cos(angle)],
                y=[0],
                z=[label_dist * np.sin(angle)],
                mode='text',
                text=[f'{angle_deg}°'],
                textposition='middle center',
                textfont=dict(size=8, color='red'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add radius labels on phi plane
        for grid_idx in range(1, num_grid_circles + 1):
            radius_frac = grid_idx / num_grid_circles
            fig3.add_trace(go.Scatter3d(
                x=[radius_frac + 0.05],
                y=[0],
                z=[0],
                mode='text',
                text=[f'r={radius_frac:.1f}'],
                textposition='middle center',
                textfont=dict(size=8, color='red'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add center point
        fig3.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(size=8, color='black'),
            name='Center',
            hoverinfo='skip'
        ))
        
        fig3.update_layout(
            title='3. Spherical Surface (θ and φ angles)',
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                yaxis=dict(range=[-1.5, 1.5], showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                zaxis=dict(range=[-1.5, 1.5], showgrid=False, showbackground=False, showticklabels=False, zeroline=False, title=''),
                aspectmode='cube'
            ),
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True
        )
        
        return [json.dumps(fig1.to_dict(), cls=NumpyEncoder), json.dumps(fig2.to_dict(), cls=NumpyEncoder), json.dumps(fig3.to_dict(), cls=NumpyEncoder)], None
    
    except Exception as e:
        return None, str(e)

def plot_3d_tessellation_transformation_data(expression, x_range=(-3, 3), y_range=(-3, 3), num_points=30, parameters=None):
    """
    3D tessellation transformation where 3D coordinates (X, Y, Z) are transformed to 2D 
    using 120° angular separation (true transformation).
    
    Transformation formulas:
    - screen_x = x - y*0.5 - z*0.5
    - screen_y = y*sqrt(3)/2 - z*sqrt(3)/2
    
    This creates an oblique triangular coordinate system where axes are 120° apart.
    """
    try:
        # Generate x, y mesh
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        
        # Filter out values too close to zero to avoid singularities
        x = x[np.abs(x) > 0.05]
        y = y[np.abs(y) > 0.05]
        
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function to get Z
        Z = safe_eval(expression, x=X, y=Y, parameters=parameters)
        if Z is None:
            raise ValueError("Function evaluation failed")
        if isinstance(Z, (int, float)):
            Z = np.full_like(X, Z, dtype=float)
        
        # Handle complex numbers
        if np.iscomplexobj(Z):
            Z = np.real(Z)
        
        # Filter out NaN and inf values
        valid_mask = np.isfinite(Z)
        if not np.any(valid_mask):
            raise ValueError("Function produced no valid values in the given range")
        
        Z = np.asarray(Z, dtype=float)
        Z[~valid_mask] = np.nanmedian(Z[valid_mask]) if np.any(valid_mask) else 0
        
        def clean_array_for_json(arr):
            """Convert numpy array to list, replacing inf/nan with safe values"""
            arr = np.asarray(arr, dtype=float)
            arr = arr.copy()
            mask = ~np.isfinite(arr)
            arr[mask] = 0
            return arr.tolist()
        
        X_list = clean_array_for_json(X)
        Y_list = clean_array_for_json(Y)
        Z_list = clean_array_for_json(Z)
        
        # ===== PLOT 1: Original 3D Cartesian (Surface) =====
        plot1_traces = [{
            'x': X_list,
            'y': Y_list,
            'z': Z_list,
            'type': 'surface',
            'colorscale': 'viridis',
            'name': 'Surface'
        }]
        layout1 = {
            'title': '1. Original (Cartesian X-Y-Z)',
            'scene': {
                'xaxis': {'title': 'x'},
                'yaxis': {'title': 'y'},
                'zaxis': {'title': 'z'}
            },
            'autosize': True,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
        
        # ===== PLOT 2: 2D Tessellation Transformation with Triangular Grid =====
        # True tessellation transformation
        # screen_x = x - y*0.5 - z*0.5
        # screen_y = y*sqrt(3)/2 - z*sqrt(3)/2
        sqrt3_2 = np.sqrt(3) / 2.0
        tes_x = X - Y * 0.5 - Z * 0.5
        tes_y = Y * sqrt3_2 - Z * sqrt3_2
        
        plot2_traces = []
        
        # Create triangular grid by drawing lines parallel to each axis
        num_grid_lines = 8
        
        # Get data ranges
        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]
        z_min, z_max = y_min, y_max  # Z range similar to Y
        
        # 1. Lines parallel to X-axis (constant Y and Z, varying X)
        for y_val in np.linspace(y_min, y_max, num_grid_lines):
            for z_val in np.linspace(z_min, z_max, num_grid_lines):
                x_line = np.linspace(x_min, x_max, 50)
                tes_x_line = x_line - y_val * 0.5 - z_val * 0.5
                tes_y_line = np.full_like(x_line, y_val * sqrt3_2 - z_val * sqrt3_2)
                
                plot2_traces.append({
                    'x': tes_x_line.tolist(),
                    'y': tes_y_line.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'color': 'rgba(180, 180, 180, 0.25)', 'width': 0.5},
                    'showlegend': False,
                    'hoverinfo': 'skip'
                })
        
        # 2. Lines parallel to Y-axis (constant X and Z, varying Y)
        for x_val in np.linspace(x_min, x_max, num_grid_lines):
            for z_val in np.linspace(z_min, z_max, num_grid_lines):
                y_line = np.linspace(y_min, y_max, 50)
                tes_x_line = x_val - y_line * 0.5 - z_val * 0.5
                tes_y_line = y_line * sqrt3_2 - z_val * sqrt3_2
                
                plot2_traces.append({
                    'x': tes_x_line.tolist(),
                    'y': tes_y_line.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'color': 'rgba(180, 180, 180, 0.25)', 'width': 0.5},
                    'showlegend': False,
                    'hoverinfo': 'skip'
                })
        
        # 3. Lines parallel to Z-axis (constant X and Y, varying Z)
        for x_val in np.linspace(x_min, x_max, num_grid_lines):
            for y_val in np.linspace(y_min, y_max, num_grid_lines):
                z_line = np.linspace(z_min, z_max, 50)
                tes_x_line = x_val - y_val * 0.5 - z_line * 0.5
                tes_y_line = y_val * sqrt3_2 - z_line * sqrt3_2
                
                plot2_traces.append({
                    'x': tes_x_line.tolist(),
                    'y': tes_y_line.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'line': {'color': 'rgba(180, 180, 180, 0.25)', 'width': 0.5},
                    'showlegend': False,
                    'hoverinfo': 'skip'
                })
        
        # Overlay the surface data as colored scatter points
        tes_x_flat = tes_x.flatten().tolist()
        tes_y_flat = tes_y.flatten().tolist()
        z_flat = Z.flatten().tolist()
        
        plot2_traces.append({
            'x': tes_x_flat,
            'y': tes_y_flat,
            'type': 'scatter',
            'mode': 'markers',
            'marker': {
                'size': 4,
                'color': z_flat,
                'colorscale': 'viridis',
                'showscale': True,
                'colorbar': {'title': 'z'},
                'opacity': 0.8
            },
            'name': 'Surface',
            'hovertemplate': 'Z: %{marker.color:.2f}<extra></extra>'
        })
        
        # Draw the three axes at origin
        x_max_range = np.max(np.abs(tes_x))
        y_max_range = np.max(np.abs(tes_y))
        z_max_range = np.max(np.abs(Z))
        
        origin_x = 0.0
        origin_y = 0.0
        
        # X-axis endpoint (red) - positive X direction
        x_ax_tes_x = x_max_range
        x_ax_tes_y = 0.0
        
        # Y-axis endpoint (green) - 120° from X (positive Y direction)
        y_ax_tes_x = -y_max_range * 0.5
        y_ax_tes_y = y_max_range * sqrt3_2
        
        # Z-axis endpoint (blue) - 240° from X (positive Z direction)
        z_ax_tes_x = -z_max_range * 0.5
        z_ax_tes_y = -z_max_range * sqrt3_2
        
        # Add axis lines with labels
        plot2_traces.extend([
            {
                'x': [origin_x, x_ax_tes_x],
                'y': [origin_y, x_ax_tes_y],
                'type': 'scatter',
                'mode': 'lines+text',
                'line': {'color': '#e63946', 'width': 3},
                'text': ['O', 'X'],
                'textposition': ['bottom left', 'top right'],
                'textfont': {'size': 12, 'color': '#e63946'},
                'showlegend': False,
                'hoverinfo': 'skip'
            },
            {
                'x': [origin_x, y_ax_tes_x],
                'y': [origin_y, y_ax_tes_y],
                'type': 'scatter',
                'mode': 'lines+text',
                'line': {'color': '#1d7fe8', 'width': 3},
                'text': ['', 'Y'],
                'textposition': ['', 'top left'],
                'textfont': {'size': 12, 'color': '#1d7fe8'},
                'showlegend': False,
                'hoverinfo': 'skip'
            },
            {
                'x': [origin_x, z_ax_tes_x],
                'y': [origin_y, z_ax_tes_y],
                'type': 'scatter',
                'mode': 'lines+text',
                'line': {'color': '#2ed573', 'width': 3},
                'text': ['', 'Z'],
                'textposition': ['', 'bottom left'],
                'textfont': {'size': 12, 'color': '#2ed573'},
                'showlegend': False,
                'hoverinfo': 'skip'
            }
        ])
        
        tes_lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.3
        layout2 = {
            'title': '2. Tessellation Transformation (120° Angular Separation)',
            'xaxis': {
                'title': '',
                'range': [-tes_lim, tes_lim],
                'scaleanchor': 'y',
                'scaleratio': 1,
                'showgrid': False,
                'zeroline': False,
            },
            'yaxis': {
                'title': '',
                'range': [-tes_lim, tes_lim],
                'showgrid': False,
                'zeroline': False,
            },
            'autosize': True,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50},
            'plot_bgcolor': 'white'
        }
        
        # ===== PLOT 3: Spherical transformation =====
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arccos(Z / np.clip(r, 1e-10, None))
        
        theta_unwrapped = np.copy(theta)
        for i in range(theta_unwrapped.shape[0]):
            theta_unwrapped[i, :] = np.unwrap(theta_unwrapped[i, :])
        
        r_min, r_max = float(np.nanmin(r)), float(np.nanmax(r))
        r_padding = (r_max - r_min) * 0.1 if r_max > r_min else 0.5
        
        theta_min, theta_max = float(np.nanmin(theta_unwrapped)), float(np.nanmax(theta_unwrapped))
        theta_padding = (theta_max - theta_min) * 0.1 if theta_max > theta_min else 0.5
        
        phi_min, phi_max = float(np.nanmin(phi)), float(np.nanmax(phi))
        phi_padding = (phi_max - phi_min) * 0.1 if phi_max > phi_min else 0.5
        
        r_list = clean_array_for_json(r)
        theta_unwrapped_list = clean_array_for_json(theta_unwrapped)
        phi_list = clean_array_for_json(phi)
        
        plot3_traces = [{
            'x': r_list,
            'y': theta_unwrapped_list,
            'z': phi_list,
            'type': 'surface',
            'colorscale': 'plasma',
            'name': 'Spherical'
        }]
        layout3 = {
            'title': '3. Spherical (r-θ-φ Cartesian)',
            'scene': {
                'xaxis': {
                    'title': 'r (radius)',
                    'range': [r_min - r_padding, r_max + r_padding]
                },
                'yaxis': {
                    'title': 'θ (azimuthal, unwrapped)',
                    'range': [theta_min - theta_padding, theta_max + theta_padding]
                },
                'zaxis': {
                    'title': 'φ (polar)',
                    'range': [phi_min - phi_padding, phi_max + phi_padding]
                }
            },
            'autosize': True,
            'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
        }
        
        return {
            'success': True,
            'plot1_traces': plot1_traces,
            'plot2_traces': plot2_traces,
            'plot3_traces': plot3_traces,
            'layout1': layout1,
            'layout2': layout2,
            'layout3': layout3
        }

    except Exception as e:
        return None, f"Error in tessellation transformation: {str(e)}"

def plot_2d_tessellation_transformation(expression, x_range=(-5, 5), y_range=(-5, 5), num_points=500, parameters=None):
    """
    Tessellation transformation of a 1D function into a triangular (120° apart) coordinate system.
    
    This mode treats the 2D function as lying in the z = 0 plane.
    The x-axis, y-axis, and z-axis are 120 degrees apart in 2D, creating a tessellated grid.
    Transformation formulas:
    - screen_x = x - y * 0.5 - z * 0.5
    - screen_y = y * sqrt(3)/2 - z * sqrt(3)/2
    
    This creates a true tessellation transformation where:
    - X-axis points right
    - Y-axis points at 120° (upper-left) 
    - Z-axis points at 240° (lower-left)
    """
    try:
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], num_points)
        x = x[np.abs(x) > 1e-9]
        
        # Evaluate function to get y
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            y = safe_eval(expression, x=x, parameters=parameters)
        
        if isinstance(y, (int, float, complex)):
            y = np.full_like(x, float(np.real(y)))
        y = np.asarray(y, dtype=np.float64)
        
        valid = np.isfinite(y) & np.isfinite(x)
        x, y = x[valid], y[valid]
        
        if len(x) == 0:
            raise ValueError("Function produced no valid values in the given range")

        # ── PLOT 1: Cartesian (Original) ──
        x_abs_max = float(np.max(np.abs(x)))
        y_abs_max = float(np.max(np.abs(y)))
        lim = max(x_abs_max, y_abs_max) * 1.2
        
        plot1_traces = [
            {'x': x.tolist(), 'y': y.tolist(),
             'type': 'scatter', 'mode': 'lines+markers',
             'name': f'y = {expression}',
             'line': {'color': '#667eea', 'width': 2},
             'marker': {'size': 3, 'opacity': 0.5}}
        ]
        layout1 = {
            'title': '1. Cartesian (Original)',
            'xaxis': {'title': 'x', 'range': [-lim, lim], 'scaleanchor': 'y', 'scaleratio': 1},
            'yaxis': {'title': 'y', 'range': [-lim, lim]},
            'autosize': True,
            'margin': {'l': 60, 'r': 50, 't': 60, 'b': 50},
            'plot_bgcolor': 'white'
        }
        
        # ── PLOT 2: 2D function transformed into triangular coordinates ──
        # The function stays 2D, but its samples are mapped into the oblique basis
        # where the three axes are 120° apart. z is conceptually 0.
        sqrt3_2 = np.sqrt(3) / 2.0
        curve_screen_x = x - 0.5 * y
        curve_screen_y = y * sqrt3_2
        
        # Create triangular grid background
        plot2_traces = []
        
        # Grid lines are drawn perpendicular to each axis direction:
        # X axis -> 90°, Y axis -> 30°, Z axis -> 150°
        num_grid = 8
        grid_y_min, grid_y_max = float(y_range[0]), float(y_range[1])
        grid_x_min, grid_x_max = float(x_range[0]), float(x_range[1])
        x_lines = np.linspace(grid_x_min, grid_x_max, num_grid)
        y_lines = np.linspace(grid_y_min, grid_y_max, num_grid)
        grid_extent_x = max(abs(grid_x_min), abs(grid_x_max), abs(np.max(curve_screen_x))) * 1.25
        grid_extent_y = max(abs(grid_y_min), abs(grid_y_max), abs(np.max(curve_screen_y))) * 1.25
        
        # Perpendicular to the X axis: vertical lines
        for y_val in y_lines:
            x_grid = np.full(100, float(y_val))
            y_grid = np.linspace(-grid_extent_y, grid_extent_y, 100)
            plot2_traces.append({
                'x': x_grid.tolist(),
                'y': y_grid.tolist(),
                'type': 'scatter', 'mode': 'lines',
                'line': {'color': 'rgba(180, 180, 180, 0.6)', 'width': 0.5},
                'showlegend': False, 'hoverinfo': 'skip'
            })
        
        # Perpendicular to the Y axis: slope = +1/sqrt(3)
        diag_span = np.linspace(-grid_extent_x, grid_extent_x, 100)
        slope_pos = 1.0 / np.sqrt(3.0)
        slope_neg = -1.0 / np.sqrt(3.0)

        for offset in x_lines:
            diag_x = diag_span
            diag_y = slope_pos * diag_x + float(offset)
            plot2_traces.append({
                'x': diag_x.tolist(),
                'y': diag_y.tolist(),
                'type': 'scatter', 
                'mode': 'lines',
                'line': {'color': 'rgba(180, 180, 180, 0.6)', 'width': 0.5},
                'showlegend': False, 
                'hoverinfo': 'skip'
            })

        # Perpendicular to the Z axis: slope = -1/sqrt(3)
        for offset in x_lines:
            diag_x = diag_span
            diag_y = slope_neg * diag_x + float(offset)
            plot2_traces.append({
                'x': diag_x.tolist(),
                'y': diag_y.tolist(),
                'type': 'scatter', 
                'mode': 'lines',
                'line': {'color': 'rgba(180, 180, 180, 0.6)', 'width': 0.5},
                'showlegend': False, 
                'hoverinfo': 'skip'
            })
        
        # Draw the function curve in the triangular coordinate system
        plot2_traces.append({
            'x': curve_screen_x.tolist(),
            'y': curve_screen_y.tolist(),
            'type': 'scatter', 
            'mode': 'lines+markers',
            'name': '',
            'line': {'color': "#e1ff00", 'width': 2},
            'marker': {'size': 4, 'opacity': 0.6}
        })
        
        # Draw axes in the 120° triangular basis
        x_axis_len = max(abs(grid_x_min), abs(grid_x_max), abs(np.max(curve_screen_x))) * 1.2
        plot2_traces.append({
            'x': [0, x_axis_len],
            'y': [0, 0],
            'type': 'scatter', 'mode': 'lines+text',
            'line': {'color': '#e63946', 'width': 2},
            'text': ['', 'X'],
            'textposition': ['bottom left', 'top right'],
            'textfont': {'size': 12, 'color': '#e63946'},
            'showlegend': False, 'hoverinfo': 'skip'
        })
        
        y_axis_len = max(abs(grid_y_min), abs(grid_y_max), abs(np.max(curve_screen_y))) * 1.2
        plot2_traces.append({
            'x': [0, -0.5 * y_axis_len],
            'y': [0, sqrt3_2 * y_axis_len],
            'type': 'scatter', 'mode': 'lines+text',
            'line': {'color': '#1d7fe8', 'width': 2},
            'text': ['', 'Y'],
            'textposition': ['', 'top left'],
            'textfont': {'size': 12, 'color': '#1d7fe8'},
            'showlegend': False, 'hoverinfo': 'skip'
        })

        z_axis_len = max(abs(grid_y_min), abs(grid_y_max), abs(np.max(curve_screen_y))) * 1.1
        plot2_traces.append({
            'x': [0, -0.5 * z_axis_len],
            'y': [0, -sqrt3_2 * z_axis_len],
            'type': 'scatter', 'mode': 'lines+text',
            'line': {'color': '#2ed573', 'width': 2, 'dash': 'dash'},
            'text': ['', 'Z'],
            'textposition': ['', 'bottom left'],
            'textfont': {'size': 12, 'color': '#2ed573'},
            'showlegend': False, 'hoverinfo': 'skip'
        })
        
        # Z is conceptually zero in this mode, so show it as an origin marker only.
        plot2_traces.append({
            'x': [0, 0],
            'y': [0, 0],
            'type': 'scatter', 'mode': 'markers+text',
            'text': ['Z=0'],
            'textposition': 'top left',
            'textfont': {'size': 12, 'color': '#2ed573'},
            'showlegend': False, 'hoverinfo': 'skip'
        })
        
        tes_lim = max(grid_extent_x, grid_extent_y, abs(x_axis_len), abs(y_axis_len), abs(z_axis_len), 1.0) * 1.2
        layout2 = {
            'title': '2. Triangular Tessellation (2D function, z = 0)',
            'xaxis': {'title': '', 'range': [-tes_lim, tes_lim], 'scaleanchor': 'y', 'scaleratio': 1,
                      'showgrid': False, 'zeroline': False},
            'yaxis': {'title': '', 'range': [-tes_lim, tes_lim], 'showgrid': False, 'zeroline': False},
            'autosize': True,
            'margin': {'l': 50, 'r': 50, 't': 60, 'b': 50},
            'plot_bgcolor': 'white'
        }
        
        # ── PLOT 3: Polar representation ──
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        plot3_traces = [
            {'x': theta.tolist(), 'y': r.tolist(),
             'type': 'scatter', 'mode': 'lines+markers',
             'name': 'Polar (θ-r space)',
             'line': {'color': '#51cf66', 'width': 2},
             'marker': {'size': 3, 'opacity': 0.5}}
        ]
        layout3 = {
            'title': '3. Polar Space (θ-r)',
            'xaxis': {'title': 'θ (radians)', 'range': [-np.pi, np.pi]},
            'yaxis': {'title': 'r (radius)', 'range': [0, np.max(r) * 1.1]},
            'autosize': True,
            'margin': {'l': 60, 'r': 50, 't': 60, 'b': 50},
            'plot_bgcolor': 'white'
        }
        
        return {
            'success': True,
            'plot1_traces': plot1_traces,
            'plot2_traces': plot2_traces,
            'plot3_traces': plot3_traces,
            'layout1': layout1,
            'layout2': layout2,
            'layout3': layout3
        }
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def parse_range_value(value, parameters=None):
    """Parse a range value that can be a number or an expression like 'pi' or '2*pi'"""
    try:
        # Try direct float conversion first (fast path for numeric strings)
        return float(value)
    except (ValueError, TypeError):
        # If that fails, try evaluating as an expression
        try:
            result = safe_eval(str(value), parameters=parameters)
            if result is None:
                raise ValueError("expression evaluation failed")
            # Convert result to scalar if it's an array
            if isinstance(result, np.ndarray):
                result = result.item()
            return float(result)
        except Exception as e:
            raise ValueError(f"Invalid range value '{value}': {str(e)}")

@app.route('/plot', methods=['POST'])
def plot():
    data = request.json
    expression = data.get('expression', 'sin(x)')
    parameters = data.get('parameters', {})
    
    try:
        x_min = parse_range_value(data.get('x_min', -5), parameters)
        x_max = parse_range_value(data.get('x_max', 5), parameters)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)})
    
    images, error = plot_function_transformation(expression, (x_min, x_max), parameters=parameters)
    
    if error:
        return jsonify({'success': False, 'error': error})
    
    return jsonify({'success': True, 'images': images})

@app.route('/plot-data', methods=['POST'])
def plot_data():
    """Return JSON data for live plotting with Plotly"""
    try:
        data = request.json
        expression = data.get('expression', 'sin(x)')
        parameters = data.get('parameters', {})
        
        x_min = parse_range_value(data.get('x_min', -5), parameters)
        x_max = parse_range_value(data.get('x_max', 5), parameters)
        num_points = 500
        
        # Generate x values
        x = np.array(np.linspace(x_min, x_max, num_points), dtype=np.float64)
        
        # Filter out values too close to zero (to avoid singularities in 1/x**2, etc.)
        # Keep only values with |x| > 0.05 to prevent divide-by-zero and overflow
        x = x[np.abs(x) > 0.05]
        
        # Suppress NumPy warnings for overflow/underflow in exp and divide by zero
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            y = safe_eval(expression, x=x, parameters=parameters)
        
        # Ensure y is an array
        if isinstance(y, complex):
            y = np.array(np.real(y), dtype=np.float64)
        else:
            y = np.asarray(y, dtype=np.float64)
        
        # Filter out NaN and inf values
        valid_mask = np.isfinite(y) & np.isfinite(x)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) == 0:
            return jsonify({'success': False, 'error': 'Function produced no valid values'})
        
        # Get dynamic limits for 1:1 aspect ratio around origin
        # This filters to only show visually interesting region near (0,0)
        limits = get_dynamic_limits(x_valid, y_valid, square_aspect=True)
        if limits:
            view_mask = (np.abs(x_valid) <= abs(limits[1])) & (np.abs(y_valid) <= abs(limits[1]))
            x_valid = x_valid[view_mask]
            y_valid = y_valid[view_mask]
        
        if len(x_valid) == 0:
            return jsonify({'success': False, 'error': 'Function produced no valid values in visible range'})
        
        # Transform to polar coordinates
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            r = np.sqrt(x_valid**2 + y_valid**2)
            theta = np.arctan2(y_valid, x_valid)
        
        # Plot 3: For polar circular plot, create an open theta domain (0 to 2π) and evaluate function there
        theta_domain = np.linspace(0, 2 * np.pi, 500, endpoint=False)
        
        with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
            r_polar = safe_eval(expression, x=theta_domain, parameters=parameters)
        
        # Handle result type and convert to array
        if isinstance(r_polar, complex):
            r_polar = np.real(r_polar)
        elif not isinstance(r_polar, np.ndarray):
            r_polar = np.array(r_polar)
        
        # Use absolute value to show full amplitude and interpolate isolated gaps
        r_polar = interpolate_invalid_1d(np.abs(r_polar))
        
        # If no valid points, use empty trace
        if len(r_polar) == 0:
            theta_degrees = np.array([], dtype=float)
            r_polar = np.array([], dtype=float)
        else:
            # Convert theta from radians to degrees for Plotly scatterpolar
            theta_degrees = np.degrees(theta_domain)
        
        # Create traces for Plotly
        traces = []
        
        # Plot 1: Cartesian
        traces.append({
            'x': x_valid.tolist(),
            'y': y_valid.tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': f'y = {expression}',
            'line': {'color': '#667eea', 'width': 2},
            'marker': {'size': 3, 'opacity': 0.3}
        })
        
        # Plot 2: Theta-r (Cartesian)
        theta_unwrapped = np.unwrap(theta)
        traces.append({
            'x': theta_unwrapped.tolist(),
            'y': r.tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Polar (θ-r)',
            'line': {'color': '#ff6b6b', 'width': 2},
            'marker': {'size': 3, 'opacity': 0.3}
        })
        
        # Plot 3: Polar circular (evaluate function in polar domain 0 to 2π)
        plot3_traces = [{
            'r': r_polar.tolist(),
            'theta': theta_degrees.tolist(),
            'type': 'scatterpolar',
            'mode': 'lines+markers',
            'name': 'Polar (circular)',
            'line': {'color': '#51cf66', 'width': 2},
            'marker': {'size': 3, 'opacity': 0.3}
        }]
        
        # Build plot2_traces list
        plot2_traces = []
        plot2_traces = [traces[1]]
        
        # Calculate dynamic axis limits for 1:1 square aspect
        x_limit = np.max(np.abs(x_valid)) * 1.1 if len(x_valid) > 0 else 1
        y_limit = np.max(np.abs(y_valid)) * 1.1 if len(y_valid) > 0 else 1
        view_limit = max(x_limit, y_limit)
        
        # Limits for polar plot
        r_limit = np.max(r) * 1.1 if len(r) > 0 else 1
        theta_limit = np.pi
        
        return jsonify({
            'success': True,
            'plot1_traces': [traces[0]],
            'plot2_traces': plot2_traces,
            'plot3_traces': plot3_traces,
            'layout1': {
                'title': '1. Original (Cartesian)',
                'xaxis': {'title': 'x', 'range': [-view_limit, view_limit]},
                'yaxis': {'title': 'y', 'range': [-view_limit, view_limit]},
                'autosize': True,
                'hovermode': 'closest',
                'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40}
            },
            'layout2': {
                'title': '2. Polar (θ-r Cartesian)',
                'xaxis': {'title': 'θ (radians)', 'range': [-theta_limit, theta_limit]},
                'yaxis': {'title': 'r (radius)', 'range': [0, r_limit]},
                'autosize': True,
                'hovermode': 'closest',
                'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40}
            },
            'layout3': {
                'title': '3. Polar (Circular)',
                'polar': {}
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/plot-tessellation-2d', methods=['POST'])
def plot_tessellation_2d():
    """Tessellation transformation of y=f(x) into triangular (120° apart) coordinate system."""
    try:
        data = request.json
        expression = data.get('expression', 'sin(x)')
        parameters = data.get('parameters', {})
        
        try:
            x_min = parse_range_value(data.get('x_min', -5), parameters)
            x_max = parse_range_value(data.get('x_max', 5), parameters)
            y_min = parse_range_value(data.get('y_min', -5), parameters)
            y_max = parse_range_value(data.get('y_max', 5), parameters)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)})
        
        try:
            num_points = int(data.get('num_points', 500))
        except (TypeError, ValueError):
            num_points = 500
        
        if num_points < 10 or num_points > 2000:
            return jsonify({'success': False, 
                            'error': 'Number of points must be between 10 and 2000'})
        
        result = plot_2d_tessellation_transformation(
            expression, (x_min, x_max), (y_min, y_max), num_points, 
            parameters=parameters
        )
        
        if not result.get('success', False):
            return jsonify({'success': False, 'error': result.get('error', 'Unknown error')})
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/plot-tessellation-3d', methods=['POST'])
@app.route('/plot-triangle', methods=['POST'])
def plot_tessellation_3d():
    """3D to 2D tessellation transformation where X, Y, Z axes are 120 degrees apart."""
    try:
        data = request.json
        expression = data.get('expression', 'sin(x)*cos(y)')
        parameters = data.get('parameters', {})

        try:
            x_min = parse_range_value(data.get('x_min', -3), parameters)
            x_max = parse_range_value(data.get('x_max', 3), parameters)
            y_min = parse_range_value(data.get('y_min', -3), parameters)
            y_max = parse_range_value(data.get('y_max', 3), parameters)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)})

        try:
            num_points = int(data.get('num_points', 30))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Invalid num_points'})

        if num_points < 2 or num_points > 500:
            return jsonify({'success': False,
                            'error': 'Number of points must be between 2 and 500'})

        result = plot_3d_tessellation_transformation_data(
            expression, (x_min, x_max), (y_min, y_max), num_points,
            parameters=parameters
        )
        
        if isinstance(result, tuple) and result[1] is not None:
            return jsonify({'success': False, 'error': result[1]})
        
        if result is None:
            return jsonify({'success': False, 'error': 'Unknown error in tessellation transformation'})
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/plot3d', methods=['POST'])
def plot3d():
    try:
        data = request.json
        expression = data.get('expression', 'sin(x)*cos(y)')
        parameters = data.get('parameters', {})
        render_format = data.get('render_format', 'plotly')  # 'plotly' or 'images'
        
        print(f"DEBUG: /plot3d called with expression: {expression}")
        print(f"DEBUG: Render format: {render_format}")
        
        try:
            x_min = parse_range_value(data.get('x_min', -3), parameters)
            x_max = parse_range_value(data.get('x_max', 3), parameters)
            y_min = parse_range_value(data.get('y_min', -3), parameters)
            y_max = parse_range_value(data.get('y_max', 3), parameters)
        except ValueError as e:
            print(f"DEBUG: Range parsing error: {e}")
            return jsonify({'success': False, 'error': str(e)})
        
        print(f"DEBUG: Ranges - x: [{x_min}, {x_max}], y: [{y_min}, {y_max}]")
        
        # Choose which function to use based on render format
        if render_format == 'images':
            images, error = plot_3d_matplotlib(expression, (x_min, x_max), (y_min, y_max), parameters=parameters)
            print(f"DEBUG: Using matplotlib 3D rendering")
        else:
            images, error = plot_3d_function_transformation(expression, (x_min, x_max), (y_min, y_max), parameters=parameters)
            print(f"DEBUG: Using Plotly 3D rendering")
        
        if error:
            print(f"DEBUG: 3D plotting error: {error}")
            return jsonify({'success': False, 'error': error})

        assert images is not None

        print(f"DEBUG: Successfully generated {len(images)} plots")
        
        # Save images to disk for debugging
        if render_format == 'images':
            with open('plot1_3d.png', 'wb') as f:
                f.write(base64.b64decode(images[0]))
            with open('plot2_3d.png', 'wb') as f:
                f.write(base64.b64decode(images[1]))
            with open('plot3_3d.png', 'wb') as f:
                f.write(base64.b64decode(images[2]))
            print("DEBUG: Saved 3D plots to plot1_3d.png, plot2_3d.png, plot3_3d.png")
        else:
            with open('plot1_3d.json', 'w') as f:
                f.write(images[0])
            with open('plot2_3d.json', 'w') as f:
                f.write(images[1])
            with open('plot3_3d.json', 'w') as f:
                f.write(images[2])
            print("DEBUG: Saved 3D plots to plot1_3d.json, plot2_3d.json, plot3_3d.json")
        
        return jsonify({'success': True, 'images': images})
    
    except Exception as e:
        print(f"DEBUG: Unexpected error in /plot3d: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Server error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
