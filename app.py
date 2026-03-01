from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import base64
from matplotlib.figure import Figure
import os
import json
import plotly.graph_objects as go

app = Flask(__name__)

def safe_eval(expression, x=None, y=None, parameters=None):
    """Safely evaluate mathematical expression with x, y variables and optional parameters"""
    
    # Helper function for product of generator/iterable
    def safe_prod(iterable):
        """Safely compute product of iterables, handling generators of arrays"""
        result = None
        for item in iterable:
            if result is None:
                result = item
            else:
                result = result * item
        return result if result is not None else 1
    
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
        # Series & aggregation
        'sum': sum,
        'prod': safe_prod,
        'range': range,
        # Constants
        'pi': np.pi,
        'e': np.e,
    }
    try:
        # Build the globals dict with all available functions
        globals_dict = safe_dict.copy()
        
        # Add variables
        if x is not None:
            globals_dict['x'] = x
        if y is not None:
            globals_dict['y'] = y
        if parameters is not None:
            globals_dict.update(parameters)
        
        # Validate syntax first
        import ast
        ast.parse(expression, mode='eval')
        
        # Evaluate with restricted builtins (empty dict restricts access but allows NumPy)
        result = eval(expression, globals_dict, {})
        return result
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {str(e)}")

def plot_function_transformation(expression, x_range=(-5, 5), num_points=500, parameters=None):
    """Generate plots for function in Cartesian, Polar (θ-r), and Polar (circular) coordinates"""
    try:
        # Generate x values
        x = np.array(np.linspace(x_range[0], x_range[1], num_points), dtype=np.float64)
        
        # Evaluate function
        y = safe_eval(expression, x=x, parameters=parameters)
        
        # Ensure y is an array
        if isinstance(y, complex):
            y = np.array(np.real(y), dtype=np.float64)
        else:
            y = np.asarray(y, dtype=np.float64)
        
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

def plot_3d_matplotlib(expression, x_range=(-3, 3), y_range=(-3, 3), num_points=30, parameters=None):
    """Generate matplotlib 3D plots as base64 images"""
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
        
        images = []
        
        # PLOT 1: Original Cartesian 3D
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Original (Cartesian X-Y-Z)')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_base64)
        plt.close(fig)
        
        # PLOT 2: Spherical coordinates as Cartesian scatter
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi = np.arccos(np.clip(Z / r, -1, 1))
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(r.flatten(), theta.flatten(), phi.flatten(), c=r.flatten(), cmap='plasma', s=10)
        ax.set_xlabel('r (radius)')
        ax.set_ylabel('θ (azimuthal)')
        ax.set_zlabel('φ (polar)')
        ax.set_title('Spherical (r-θ-φ Cartesian)')
        fig.colorbar(scatter, ax=ax, label='r')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_base64)
        plt.close(fig)
        
        # PLOT 3: Data projected on sphere
        r_max = np.max(r)
        if r_max > 0:
            r_norm = r / r_max
        else:
            r_norm = r
        r_sphere = 1 + 0.3 * r_norm
        
        X_sph = r_sphere * np.sin(phi) * np.cos(theta)
        Y_sph = r_sphere * np.sin(phi) * np.sin(theta)
        Z_sph = r_sphere * np.cos(phi)
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_sph.flatten(), Y_sph.flatten(), Z_sph.flatten(), c=r.flatten(), cmap='coolwarm', s=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Spherical (Data on Sphere)')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        fig.colorbar(scatter, ax=ax, label='r')
        
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        images.append(img_base64)
        plt.close(fig)
        
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
        
        # CRITICAL: Convert numpy arrays to Python lists for JSON serialization
        X_list = X.tolist()
        Y_list = Y.tolist()
        Z_list = Z.tolist()
        r_list = r.tolist()
        theta_list = theta.tolist()
        phi_list = phi.tolist()
        
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
        fig2 = go.Figure(data=[go.Surface(x=r_list, y=theta_list, z=phi_list, colorscale='plasma', name='Surface')])
        fig2.update_layout(
            title='2. Spherical (r-θ-φ Cartesian)',
            scene=dict(
                xaxis_title='r (radius)',
                yaxis_title='θ (azimuthal)',
                zaxis_title='φ (polar)'
            ),
            autosize=True,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # ===== PLOT 3: 3D Spherical Surface =====
        # Create a 3D surface in spherical coordinates where radius varies with both theta and phi
        
        fig3 = go.Figure()
        
        # Create a grid of theta and phi values
        theta_grid = np.linspace(-np.pi, np.pi, 60)
        phi_grid = np.linspace(0.1, np.pi - 0.1, 40)  # Avoid singularities at 0 and π
        THETA, PHI = np.meshgrid(theta_grid, phi_grid)
        
        # Evaluate function at each (theta, phi) point
        R = safe_eval(expression, x=THETA, y=PHI, parameters=parameters)
        if isinstance(R, (int, float)):
            R = np.full_like(THETA, R)
        R = np.abs(np.asarray(R, dtype=float))
        
        # Normalize radius for visualization (prevent too large/small values)
        r_max = np.max(np.abs(R)) + 1e-6
        r_norm = np.abs(R) / r_max
        
        # Convert spherical coordinates to Cartesian
        X_sphere = r_norm * np.sin(PHI) * np.cos(THETA)
        Y_sphere = r_norm * np.sin(PHI) * np.sin(THETA)
        Z_sphere = r_norm * np.cos(PHI)
        
        # Create the 3D surface
        fig3.add_trace(go.Surface(
            x=X_sphere.tolist(),
            y=Y_sphere.tolist(),
            z=Z_sphere.tolist(),
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
        
        return [json.dumps(fig1.to_dict()), json.dumps(fig2.to_dict()), json.dumps(fig3.to_dict())], None
    
    except Exception as e:
        return None, str(e)

def parse_range_value(value, parameters=None):
    """Parse a range value that can be a number or an expression like 'pi' or '2*pi'"""
    try:
        # Try direct float conversion first (fast path for numeric strings)
        return float(value)
    except (ValueError, TypeError):
        # If that fails, try evaluating as an expression
        try:
            result = safe_eval(str(value), parameters=parameters)
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
        
        # Evaluate function
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
        
        # Transform to polar coordinates
        r = np.sqrt(x_valid**2 + y_valid**2)
        theta = np.arctan2(y_valid, x_valid)
        
        # Plot 3: For polar circular plot, create a full theta domain (0 to 2π) and evaluate function there
        theta_domain = np.linspace(0, 2 * np.pi, 500)
        r_polar = safe_eval(expression, x=theta_domain, parameters=parameters)
        
        # Handle result type and convert to array
        if isinstance(r_polar, complex):
            r_polar = np.real(r_polar)
        elif not isinstance(r_polar, np.ndarray):
            r_polar = np.array(r_polar)
        
        # Use absolute value to show full amplitude
        r_polar = np.abs(r_polar)
        
        # Convert theta from radians to degrees for Plotly scatterpolar (expects degrees by default)
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
        if len(discontinuities) == 0:
            plot2_traces = [traces[1]]
        else:
            # traces[1:] contains all the split traces for plot 2
            plot2_traces = traces[1:]
        
        return jsonify({
            'success': True,
            'plot1_traces': [traces[0]],
            'plot2_traces': plot2_traces,
            'plot3_traces': plot3_traces,
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
