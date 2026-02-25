import numpy as np
import matplotlib.pyplot as plt

class PolarTransformer:
    """Transform 2D and 3D objects to polar/spherical coordinates"""
    
    @staticmethod
    def cartesian_to_polar_2d(x, y):
        """Convert 2D Cartesian to polar coordinates"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    
    @staticmethod
    def cartesian_to_spherical(x, y, z):
        """Convert 3D Cartesian to spherical coordinates"""
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # azimuthal angle
        phi = np.arccos(z / np.clip(r, 1e-10, None))  # polar angle
        return r, theta, phi


class Shape2D:
    """Generate 2D shapes for transformation"""
    
    @staticmethod
    def rectangle(width=3.0, height=1.5, points=100):
        """Generate a rectangle - will transform into spiral-like shape in polar"""
        # Bottom
        x1 = np.linspace(-width/2, width/2, points//4)
        y1 = np.full(points//4, -height/2)
        # Right
        x2 = np.full(points//4, width/2)
        y2 = np.linspace(-height/2, height/2, points//4)
        # Top
        x3 = np.linspace(width/2, -width/2, points//4)
        y3 = np.full(points//4, height/2)
        # Left
        x4 = np.full(points//4, -width/2)
        y4 = np.linspace(height/2, -height/2, points//4)
        
        x = np.concatenate([x1, x2, x3, x4])
        y = np.concatenate([y1, y2, y3, y4])
        return x, y
    
    @staticmethod
    def grid(x_range=3.0, y_range=2.0, spacing=0.5):
        """Generate a grid pattern - shows polar distortion clearly"""
        x, y = [], []
        
        # Horizontal lines
        y_vals = np.arange(-y_range/2, y_range/2 + spacing, spacing)
        for y_val in y_vals:
            x_line = np.linspace(-x_range/2, x_range/2, 30)
            x.extend(x_line)
            y.extend(np.full(30, y_val))
        
        # Vertical lines
        x_vals = np.arange(-x_range/2, x_range/2 + spacing, spacing)
        for x_val in x_vals:
            y_line = np.linspace(-y_range/2, y_range/2, 30)
            x.extend(np.full(30, x_val))
            y.extend(y_line)
        
        return np.array(x), np.array(y)
    
    @staticmethod
    def line_vertical(x_pos=1.0, y_range=3.0, points=100):
        """Generate a vertical line - will become radial line in polar"""
        x = np.full(points, x_pos)
        y = np.linspace(-y_range/2, y_range/2, points)
        return x, y


class Shape3D:
    """Generate 3D shapes for transformation"""
    
    @staticmethod
    def rectangular_box(width=2.0, height=1.5, depth=1.0, points=25):
        """Generate a rectangular box - will show interesting spherical transformation"""
        x, y, z = [], [], []
        
        w, h, d = width/2, height/2, depth/2
        u = np.linspace(-1, 1, points)
        v = np.linspace(-1, 1, points)
        uu, vv = np.meshgrid(u, v)
        
        # Front face (z = d)
        x.append((uu * w).flatten())
        y.append((vv * h).flatten())
        z.append(np.full(uu.size, d))
        
        # Back face (z = -d)  
        x.append((uu * w).flatten())
        y.append((vv * h).flatten())
        z.append(np.full(uu.size, -d))
        
        # Top face (y = h)
        xx, zz = np.meshgrid(u, v)
        x.append((xx * w).flatten())
        y.append(np.full(xx.size, h))
        z.append((zz * d).flatten())
        
        # Bottom face (y = -h)
        x.append((xx * w).flatten())
        y.append(np.full(xx.size, -h))
        z.append((zz * d).flatten())
        
        return np.concatenate(x), np.concatenate(y), np.concatenate(z)
    
    @staticmethod
    def vertical_plane(width=2.0, height=2.0, points=30):
        """Generate a vertical plane - will show radial pattern in spherical"""
        u = np.linspace(-width/2, width/2, points)
        v = np.linspace(-height/2, height/2, points)
        uu, vv = np.meshgrid(u, v)
        
        x = uu
        y = np.full_like(uu, 1.0)  # Fixed y distance
        z = vv
        
        return x.flatten(), y.flatten(), z.flatten()
    
    @staticmethod
    def spiral_3d(max_radius=1.5, height=3.0, points=200):
        """Generate a 3D spiral - interesting transformation"""
        t = np.linspace(0, 4*np.pi, points)
        x = max_radius * np.cos(t)
        y = max_radius * np.sin(t)
        z = np.linspace(0, height, points) - height/2
        return x, y, z


def plot_2d_transformation(x, y, title="2D Transformation to Polar Coordinates"):
    """Plot 2D shape transformed to polar coordinates"""
    r, theta = PolarTransformer.cartesian_to_polar_2d(x, y)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original Cartesian
    axes[0].plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
    axes[0].scatter(x, y, s=10, alpha=0.3, color='blue')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Original (Cartesian X-Y)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Transformed to Polar coordinates - plot r vs theta as Cartesian axes
    axes[1].plot(theta, r, 'r-', linewidth=1.5, alpha=0.7)
    axes[1].scatter(theta, r, s=10, alpha=0.3, color='red')
    axes[1].set_xlabel('θ (angle in radians)')
    axes[1].set_ylabel('r (radius)')
    axes[1].set_title('Transformed to Polar (θ-r space)')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_3d_transformation(x, y, z, title="3D Transformation to Spherical Coordinates"):
    """Plot 3D shape transformed to spherical coordinates"""
    r, theta, phi = PolarTransformer.cartesian_to_spherical(x, y, z)
    
    fig = plt.figure(figsize=(14, 6))
    
    # Original Cartesian
    ax1 = fig.add_subplot(121, projection='3d')
    # scatter1 = ax1.scatter(x, y, z, c='blue', s=5, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original (Cartesian X-Y-Z)')
    
    # Spherical transformation - plot r, theta, phi as Cartesian axes
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(r, theta, phi, c=r, cmap='plasma', s=5, alpha=0.5)
    ax2.set_xlabel('r (radius)')
    ax2.set_ylabel('θ (azimuthal angle)')
    ax2.set_zlabel('φ (polar angle)')
    ax2.set_title('Transformed to Spherical (r-θ-φ space)')
    fig.colorbar(scatter2, ax=ax2, label='r value')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Main function to demonstrate transformations"""
    
    import os
    
    # Create output directory for images
    output_dir = "polar_transform_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Generating 2D transformations...")
    
    # Rectangle - shows how edges transform
    x_rect, y_rect = Shape2D.rectangle(width=3.0, height=1.5, points=200)
    fig1 = plot_2d_transformation(x_rect, y_rect, "Rectangle → Polar Coordinates")
    fig1.savefig(os.path.join(output_dir, "01_rectangle_to_polar.png"), dpi=150, bbox_inches='tight')
    print("Saved: 01_rectangle_to_polar.png")
    plt.close(fig1)
    
    # Grid - shows coordinate system distortion
    x_grid, y_grid = Shape2D.grid(x_range=3.0, y_range=2.0, spacing=0.5)
    fig2 = plot_2d_transformation(x_grid, y_grid, "Grid → Polar Coordinates")
    fig2.savefig(os.path.join(output_dir, "02_grid_to_polar.png"), dpi=150, bbox_inches='tight')
    print("Saved: 02_grid_to_polar.png")
    plt.close(fig2)
    
    # Vertical line - becomes radial in polar
    x_line, y_line = Shape2D.line_vertical(x_pos=0, y_range=3.0, points=100)
    fig3 = plot_2d_transformation(x_line, y_line, "Vertical Line → Polar Coordinates")
    fig3.savefig(os.path.join(output_dir, "03_vertical_line_to_polar.png"), dpi=150, bbox_inches='tight')
    print("Saved: 03_vertical_line_to_polar.png")
    plt.close(fig3)
    
    print("Generating 3D transformations...")
    
    # Rectangular box
    x_box, y_box, z_box = Shape3D.rectangular_box(width=2.0, height=1.5, depth=1.0, points=20)
    fig4 = plot_3d_transformation(x_box, y_box, z_box, "Rectangular Box → Spherical Coordinates")
    fig4.savefig(os.path.join(output_dir, "04_rectangular_box_to_spherical.png"), dpi=150, bbox_inches='tight')
    print("Saved: 04_rectangular_box_to_spherical.png")
    plt.close(fig4)
    
    # Vertical plane
    x_plane, y_plane, z_plane = Shape3D.vertical_plane(width=2.0, height=2.0, points=30)
    fig5 = plot_3d_transformation(x_plane, y_plane, z_plane, "Vertical Plane → Spherical Coordinates")
    fig5.savefig(os.path.join(output_dir, "05_vertical_plane_to_spherical.png"), dpi=150, bbox_inches='tight')
    print("Saved: 05_vertical_plane_to_spherical.png")
    plt.close(fig5)
    
    # 3D spiral
    x_spiral, y_spiral, z_spiral = Shape3D.spiral_3d(max_radius=1.5, height=3.0, points=150)
    fig6 = plot_3d_transformation(x_spiral, y_spiral, z_spiral, "3D Spiral → Spherical Coordinates")
    fig6.savefig(os.path.join(output_dir, "06_spiral_3d_to_spherical.png"), dpi=150, bbox_inches='tight')
    print("Saved: 06_spiral_3d_to_spherical.png")
    plt.close(fig6)
    
    print(f"\nAll images saved to '{output_dir}' directory")


if __name__ == "__main__":
    main()
