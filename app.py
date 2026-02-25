from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from io import BytesIO
import base64
from matplotlib.figure import Figure
import os
import plotly.graph_objects as go

app = Flask(__name__)

def safe_eval(expression, x=None, y=None, parameters=None):
    """Safely evaluate mathematical expression with x, y variables and optional parameters"""
    # Create a safe namespace with common math functions
    safe_dict = {
        # Trigonometric functions
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
        # Constants
        'pi': np.pi,
        'e': np.e,
    }
    try:
        # Build the namespace with available variables
        namespace = {**safe_dict}
        if x is not None:
            namespace['x'] = x
        if y is not None:
            namespace['y'] = y
        if parameters is not None:
            namespace.update(parameters)
        
        result = eval(expression, {"__builtins__": {}}, namespace)
        return result
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def plot_function_transformation(expression, x_range=(-5, 5), num_points=500, parameters=None):
    """Generate plots for function in Cartesian, Polar (θ-r), and Polar (circular) coordinates"""
    try:
        # Generate x values
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Evaluate function
        y = safe_eval(expression, x=x, parameters=parameters)
        
        # Handle complex numbers or invalid results
        if isinstance(y, complex):
            y = np.real(y)
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Filter out NaN and inf values
        valid_mask = np.isfinite(y) & np.isfinite(x)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) == 0:
            raise ValueError("Function produced no valid values in the given range")
        
        # Transform to polar coordinates
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        
        # Find discontinuities in theta (where it wraps around ±π)
        theta_diff = np.abs(np.diff(theta))
        discontinuities = np.where(theta_diff > np.pi/2)[0]
        
        # ===== PLOT 1: Cartesian ====
        fig1 = Figure(figsize=(6, 6), dpi=100)
        ax1 = fig1.add_subplot(111)
        ax1.plot(x, y, 'b-', linewidth=2, label=f'y = {expression}')
        ax1.scatter(x, y, s=5, alpha=0.3, color='blue')
        
        # Accentuate 0,0 axes
        ax1.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
        ax1.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
        
        ax1.set_xlabel('x', fontsize=11)
        ax1.set_ylabel('y', fontsize=11)
        ax1.set_title('1. Original (Cartesian X-Y)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        
        img_buffer1 = BytesIO()
        fig1.savefig(img_buffer1, format='png', bbox_inches='tight')
        img_buffer1.seek(0)
        img1_base64 = base64.b64encode(img_buffer1.getvalue()).decode()
        plt.close(fig1)
        
        # ===== PLOT 2: Polar (θ-r Cartesian space) =====
        fig2 = Figure(figsize=(6, 6), dpi=100)
        ax2 = fig2.add_subplot(111)
        
        # Plot segments, breaking at discontinuities
        if len(discontinuities) == 0:
            # No discontinuities, plot normally
            ax2.plot(theta, r, 'r-', linewidth=2, label='Transformed')
            ax2.scatter(theta, r, s=5, alpha=0.3, color='red')
        else:
            # Plot with breaks at discontinuities
            prev_idx = 0
            for disc_idx in discontinuities:
                # Plot solid line for the continuous segment
                ax2.plot(theta[prev_idx:disc_idx+1], r[prev_idx:disc_idx+1], 'r-', linewidth=2)
                ax2.scatter(theta[prev_idx:disc_idx+1], r[prev_idx:disc_idx+1], s=5, alpha=0.3, color='red')
                prev_idx = disc_idx + 1
            
            # Plot the last segment
            ax2.plot(theta[prev_idx:], r[prev_idx:], 'r-', linewidth=2)
            ax2.scatter(theta[prev_idx:], r[prev_idx:], s=5, alpha=0.3, color='red')
            ax2.plot([], [], 'r-', linewidth=2, label='Transformed (solid)')
        
        # Accentuate 0,0 axes
        ax2.axhline(y=0, color='black', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=0, color='black', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('θ (radians)', fontsize=11)
        ax2.set_ylabel('r (radius)', fontsize=11)
        ax2.set_title('2. Polar (θ-r Cartesian)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        
        img_buffer2 = BytesIO()
        fig2.savefig(img_buffer2, format='png', bbox_inches='tight')
        img_buffer2.seek(0)
        img2_base64 = base64.b64encode(img_buffer2.getvalue()).decode()
        plt.close(fig2)
        
        # ===== PLOT 3: Proper polar plot (circular representation) =====
        fig3 = Figure(figsize=(6, 6), dpi=100)
        ax3 = fig3.add_subplot(111, projection='polar')
        
        # For polar circular plot: interpret x as angles and y as radius
        # Map x range to full circle [0, 2π]
        theta_polar = (x - x.min()) / (x.max() - x.min()) * 2 * np.pi
        r_polar = np.abs(y)  # Use absolute value of function as radius
        
        # Plot directly in polar coordinates
        ax3.plot(theta_polar, r_polar, 'g-', linewidth=2, label='r = |f(θ)|')
        ax3.scatter(theta_polar, r_polar, s=5, alpha=0.3, color='green')
        
        ax3.set_title('3. Polar (Function as Radius)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        
        img_buffer3 = BytesIO()
        fig3.savefig(img_buffer3, format='png', bbox_inches='tight')
        img_buffer3.seek(0)
        img3_base64 = base64.b64encode(img_buffer3.getvalue()).decode()
        plt.close(fig3)
        
        return [img1_base64, img2_base64, img3_base64], None
    
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
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function with both x and y
        Z = safe_eval(expression, x=X, y=Y, parameters=parameters)
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
        
        # ===== PLOT 1: Original Cartesian 3D (Surface) =====
        fig1 = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis', name='Surface')])
        fig1.update_layout(
            title='1. Original (Cartesian X-Y-Z)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z'
            ),
            width=600,
            height=600
        )
        
        # ===== PLOT 2: Spherical as Cartesian axes (r, theta, phi) =====
        fig2 = go.Figure(data=[go.Scatter3d(
            x=r.flatten(),
            y=theta.flatten(),
            z=phi.flatten(),
            mode='markers',
            marker=dict(
                size=3,
                color=r.flatten(),
                colorscale='plasma',
                showscale=True,
                colorbar=dict(title="r")
            ),
            name='Data'
        )])
        fig2.update_layout(
            title='2. Spherical (r-θ-φ Cartesian)',
            scene=dict(
                xaxis_title='r (radius)',
                yaxis_title='θ (azimuthal)',
                zaxis_title='φ (polar)'
            ),
            width=600,
            height=600
        )
        
        # ===== PLOT 3: Data projected on sphere =====
        r_max = np.max(r)
        if r_max > 0:
            r_norm = r / r_max
        else:
            r_norm = r
        r_sphere = 1 + 0.3 * r_norm
        
        X_sph = r_sphere * np.sin(phi) * np.cos(theta)
        Y_sph = r_sphere * np.sin(phi) * np.sin(theta)
        Z_sph = r_sphere * np.cos(phi)
        
        fig3 = go.Figure(data=[
            go.Scatter3d(
                x=X_sph.flatten(),
                y=Y_sph.flatten(),
                z=Z_sph.flatten(),
                mode='markers',
                marker=dict(
                    size=3,
                    color=r.flatten(),
                    colorscale='icefire',
                    showscale=True,
                    colorbar=dict(title="r")
                ),
                name='Data'
            )
        ])
        
        # Add sphere wireframe to plot 3
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig3.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.15,
            colorscale='greys',
            showscale=False,
            name='Sphere'
        ))
        
        fig3.update_layout(
            title='3. Spherical (Data on Sphere)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                xaxis=dict(range=[-1.5, 1.5]),
                yaxis=dict(range=[-1.5, 1.5]),
                zaxis=dict(range=[-1.5, 1.5])
            ),
            width=600,
            height=600
        )
        
        return [fig1.to_json(), fig2.to_json(), fig3.to_json()], None
    
    except Exception as e:
        return None, str(e)

@app.route('/plot', methods=['POST'])
def plot():
    data = request.json
    expression = data.get('expression', 'sin(x)')
    x_min = float(data.get('x_min', -5))
    x_max = float(data.get('x_max', 5))
    parameters = data.get('parameters', {})
    
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
        x_min = float(data.get('x_min', -5))
        x_max = float(data.get('x_max', 5))
        parameters = data.get('parameters', {})
        num_points = 500
        
        # Generate x values
        x = np.linspace(x_min, x_max, num_points)
        
        # Evaluate function
        y = safe_eval(expression, x=x, parameters=parameters)
        
        # Handle complex numbers or invalid results
        if isinstance(y, complex):
            y = np.real(y)
        elif not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Filter out NaN and inf values
        valid_mask = np.isfinite(y) & np.isfinite(x)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) == 0:
            return jsonify({'success': False, 'error': 'Function produced no valid values'})
        
        # Transform to polar coordinates
        r = np.sqrt(x_valid**2 + y_valid**2)
        theta = np.arctan2(y_valid, x_valid)
        
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
        # Detect discontinuities
        theta_diff = np.abs(np.diff(theta))
        discontinuities = np.where(theta_diff > np.pi/2)[0]
        
        if len(discontinuities) == 0:
            traces.append({
                'x': theta.tolist(),
                'y': r.tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Polar (θ-r)',
                'line': {'color': '#ff6b6b', 'width': 2},
                'marker': {'size': 3, 'opacity': 0.3}
            })
        else:
            # Split at discontinuities
            prev_idx = 0
            for i, disc_idx in enumerate(discontinuities):
                traces.append({
                    'x': theta[prev_idx:disc_idx+1].tolist(),
                    'y': r[prev_idx:disc_idx+1].tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': f'Polar (θ-r)' if i == 0 else '',
                    'line': {'color': '#ff6b6b', 'width': 2},
                    'marker': {'size': 3, 'opacity': 0.3},
                    'showlegend': i == 0
                })
                prev_idx = disc_idx + 1
            
            # Last segment
            traces.append({
                'x': theta[prev_idx:].tolist(),
                'y': r[prev_idx:].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'line': {'color': '#ff6b6b', 'width': 2},
                'marker': {'size': 3, 'opacity': 0.3},
                'showlegend': False
            })
        
        # Plot 3: Polar circular
        x_normalized = (x_valid - x_valid.min()) / (x_valid.max() - x_valid.min()) * 2 * np.pi
        r_func = np.abs(y_valid)
        
        traces.append({
            'r': r_func.tolist(),
            'theta': x_normalized.tolist(),
            'type': 'scatterpolar',
            'mode': 'lines+markers',
            'name': 'Polar (circular)',
            'line': {'color': '#51cf66', 'width': 2},
            'marker': {'size': 3, 'opacity': 0.3}
        })
        
        return jsonify({
            'success': True,
            'traces': traces,
            'layout1': {
                'title': '1. Original (Cartesian)',
                'xaxis': {'title': 'x'},
                'yaxis': {'title': 'y'},
                'autosize': True,
                'hovermode': 'closest',
                'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40}
            },
            'layout2': {
                'title': '2. Polar (θ-r Cartesian)',
                'xaxis': {'title': 'θ (radians)'},
                'yaxis': {'title': 'r (radius)'},
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

@app.route('/plot3d', methods=['POST'])
def plot3d():
    data = request.json
    expression = data.get('expression', 'sin(x)*cos(y)')
    x_min = float(data.get('x_min', -3))
    x_max = float(data.get('x_max', 3))
    y_min = float(data.get('y_min', -3))
    y_max = float(data.get('y_max', 3))
    parameters = data.get('parameters', {})
    
    images, error = plot_3d_function_transformation(expression, (x_min, x_max), (y_min, y_max), parameters=parameters)
    
    if error:
        return jsonify({'success': False, 'error': error})
    
    return jsonify({'success': True, 'images': images})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
