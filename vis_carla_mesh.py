import open3d as o3d
import numpy as np

def visualize_ply_with_axes(file_path):
    try:
        # 读取PLY文件
        mesh = o3d.io.read_triangle_mesh(file_path)
        

        # 预处理
        mesh.compute_vertex_normals()  # 计算法线
        # mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 设置默认颜色
        
        num_faces = len(mesh.triangles)
        # face_colors = np.random.rand(num_faces, 3)  
        # mesh.triangle_colors = o3d.utility.Vector3dVector(face_colors)

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="3D Viewer - " + file_path,
            width=1024,
            height=768
        )
        
        # 添加主模型
        vis.add_geometry(mesh)

        # 添加坐标系（X红色，Y绿色，Z蓝色）
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,  # 坐标轴长度
            origin=[0, 0, 0]  # 坐标系原点
        )
        vis.add_geometry(coordinate_frame)

        # 设置渲染选项
        render_opt = vis.get_render_option()
        render_opt.mesh_show_wireframe = True   # 显示线框
        render_opt.mesh_show_back_face = True   # 显示背面
        # render_opt.background_color = [0.1, 0.1, 0.1]  # 深灰色背景

        # 设置初始视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)  # 初始缩放级别

        # 交互式可视化
        vis.run()
        
        # 清理资源
        vis.destroy_window()

    except Exception as e:
        print(f"可视化失败: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_file = "/Users/junjiewang/Downloads/004507.ply"  # 修改为你的PLY文件路径
    visualize_ply_with_axes(input_file)
    