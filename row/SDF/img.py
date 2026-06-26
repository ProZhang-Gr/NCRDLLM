import os
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import re


def generate_molecule_image(sdf_path, cid, output_dir="drugimages", image_size=(400, 400)):
    """
    从SDF文件生成分子结构图像

    Args:
        sdf_path: SDF文件路径
        cid: 化合物CID
        output_dir: 图像输出目录
        image_size: 图像大小 (width, height)

    Returns:
        str: 图像文件路径，失败时返回None
    """
    try:
        # 读取SDF文件
        mol = None

        # 尝试多种方式读取分子
        try:
            mol = Chem.MolFromMolFile(sdf_path)
        except:
            try:
                mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=False)
                if mol is not None:
                    Chem.SanitizeMol(mol)
            except:
                supplier = Chem.SDMolSupplier(sdf_path)
                for m in supplier:
                    if m is not None:
                        mol = m
                        break

        if mol is None:
            print(f"  ✗ 无法读取分子: {sdf_path}")
            return None

        # 生成2D坐标（如果没有的话）
        if not mol.GetNumConformers():
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)

        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)

        # 设置图像路径
        image_path = os.path.join(output_dir, f"{cid}.png")

        # 生成分子图像
        img = Draw.MolToImage(mol, size=image_size, kekulize=True)

        # 保存图像
        img.save(image_path)

        print(f"  ✓ 生成图像: {image_path}")
        return image_path

    except Exception as e:
        print(f"  ✗ 生成图像失败: {e}")
        return None


def generate_high_quality_image(sdf_path, cid, output_dir="drugimages", image_size=(600, 600)):
    """
    生成高质量分子结构图像（使用rdMolDraw2D）

    Args:
        sdf_path: SDF文件路径
        cid: 化合物CID
        output_dir: 图像输出目录
        image_size: 图像大小 (width, height)

    Returns:
        str: 图像文件路径，失败时返回None
    """
    try:
        # 读取分子
        mol = None

        try:
            mol = Chem.MolFromMolFile(sdf_path)
        except:
            try:
                mol = Chem.MolFromMolFile(sdf_path, sanitize=False, removeHs=False)
                if mol is not None:
                    Chem.SanitizeMol(mol)
            except:
                supplier = Chem.SDMolSupplier(sdf_path)
                for m in supplier:
                    if m is not None:
                        mol = m
                        break

        if mol is None:
            print(f"  ✗ 无法读取分子: {sdf_path}")
            return None

        # 生成2D坐标
        if not mol.GetNumConformers():
            from rdkit.Chem import rdDepictor
            rdDepictor.Compute2DCoords(mol)

        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)

        # 设置图像路径
        image_path = os.path.join(output_dir, f"{cid}.png")

        # 创建高质量绘图器
        drawer = rdMolDraw2D.MolDraw2DCairo(image_size[0], image_size[1])

        # 设置绘图选项
        drawer.SetFontSize(0.8)
        drawer.SetLineWidth(2)

        # 绘制分子
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()

        # 保存图像
        with open(image_path, 'wb') as f:
            f.write(drawer.GetDrawingText())

        print(f"  ✓ 生成高质量图像: {image_path}")
        return image_path

    except Exception as e:
        print(f"  ✗ 生成高质量图像失败，尝试标准方法: {e}")
        # 回退到标准方法
        return generate_molecule_image(sdf_path, cid, output_dir, image_size)


def batch_generate_drug_images(sdf_dir, output_dir="drugimages", high_quality=True, image_size=(600, 600)):
    """
    批量生成分子结构图像

    Args:
        sdf_dir: SDF文件目录
        output_dir: 图像输出目录
        high_quality: 是否使用高质量渲染
        image_size: 图像大小

    Returns:
        dict: 处理结果统计
    """
    print("开始批量生成分子结构图像...")

    # 获取所有SDF文件
    if not os.path.exists(sdf_dir):
        print(f"✗ SDF目录不存在: {sdf_dir}")
        return None

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')]

    if not sdf_files:
        print(f"在目录 {sdf_dir} 中未找到SDF文件")
        return None

    print(f"找到 {len(sdf_files)} 个SDF文件")
    print(f"输出目录: {output_dir}")
    print(f"图像大小: {image_size}")
    print(f"高质量渲染: {high_quality}")

    # 统计变量
    success_count = 0
    failed_files = []

    # 逐个处理SDF文件
    for idx, sdf_file in enumerate(sdf_files):
        # 从文件名提取CID
        cid = sdf_file.replace('.sdf', '')

        print(f"\n处理 {idx + 1}/{len(sdf_files)}: {sdf_file} (CID: {cid})")

        # SDF文件路径
        sdf_path = os.path.join(sdf_dir, sdf_file)

        # 生成图像
        if high_quality:
            image_path = generate_high_quality_image(sdf_path, cid, output_dir, image_size)
        else:
            image_path = generate_molecule_image(sdf_path, cid, output_dir, image_size)

        if image_path:
            success_count += 1
        else:
            failed_files.append(sdf_file)

    # 统计结果
    result = {
        'total_files': len(sdf_files),
        'success_count': success_count,
        'failed_count': len(failed_files),
        'success_rate': success_count / len(sdf_files) * 100,
        'failed_files': failed_files,
        'output_dir': output_dir
    }

    return result


def generate_image_grid(image_dir="drugimages", grid_size=(5, 4), output_file="drug_grid.png"):
    """
    将多个分子图像合并为一个网格图像

    Args:
        image_dir: 图像目录
        grid_size: 网格大小 (列, 行)
        output_file: 输出文件名
    """
    try:
        from PIL import Image
        import math

        # 获取所有PNG图像
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

        if not image_files:
            print("未找到PNG图像文件")
            return

        # 取前grid_size[0] * grid_size[1]个图像
        max_images = grid_size[0] * grid_size[1]
        selected_files = image_files[:max_images]

        print(f"创建 {grid_size[0]}x{grid_size[1]} 网格，使用 {len(selected_files)} 个图像")

        # 读取第一个图像获取尺寸
        first_img = Image.open(os.path.join(image_dir, selected_files[0]))
        img_width, img_height = first_img.size

        # 创建网格图像
        grid_width = grid_size[0] * img_width
        grid_height = grid_size[1] * img_height
        grid_img = Image.new('RGB', (grid_width, grid_height), 'white')

        # 填充网格
        for i, img_file in enumerate(selected_files):
            row = i // grid_size[0]
            col = i % grid_size[0]

            img = Image.open(os.path.join(image_dir, img_file))
            x = col * img_width
            y = row * img_height
            grid_img.paste(img, (x, y))

        # 保存网格图像
        grid_path = os.path.join(image_dir, output_file)
        grid_img.save(grid_path)

        print(f"✓ 网格图像已保存: {grid_path}")

    except ImportError:
        print("需要安装PIL: pip install Pillow")
    except Exception as e:
        print(f"✗ 创建网格图像失败: {e}")


def print_image_generation_summary(result):
    """
    打印图像生成结果摘要
    """
    if result is None:
        return

    print("\n" + "=" * 60)
    print("分子结构图像生成结果摘要")
    print("=" * 60)

    print(f"总SDF文件数: {result['total_files']}")
    print(f"成功生成图像: {result['success_count']}")
    print(f"生成失败: {result['failed_count']}")
    print(f"成功率: {result['success_rate']:.1f}%")

    if result['failed_files']:
        print(f"\n生成失败的文件:")
        for filename in result['failed_files']:
            print(f"  - {filename}")
    else:
        print(f"\n🎉 所有SDF文件都成功生成了分子结构图像!")

    print(f"\n图像保存位置: {result['output_dir']}")


def verify_generated_images(output_dir="drugimages"):
    """
    验证生成的图像文件
    """
    print(f"\n验证生成的图像文件...")

    if not os.path.exists(output_dir):
        print(f"图像目录不存在: {output_dir}")
        return

    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    print(f"找到 {len(image_files)} 个PNG图像文件")

    # 检查文件大小
    total_size = 0
    for img_file in image_files[:10]:  # 检查前10个文件
        img_path = os.path.join(output_dir, img_file)
        size = os.path.getsize(img_path)
        total_size += size
        print(f"  {img_file}: {size / 1024:.1f} KB")

    if image_files:
        avg_size = total_size / min(len(image_files), 10)
        print(f"\n平均文件大小: {avg_size / 1024:.1f} KB")


# 使用示例
if __name__ == "__main__":
    # SDF文件目录
    sdf_dir = r"D:\Desktop\CDLLM\ing\row\SDF\sdf_files"

    # 输出目录
    output_dir = "drugimages"

    print("开始生成分子结构图像...")

    # 批量生成图像
    result = batch_generate_drug_images(
        sdf_dir=sdf_dir,
        output_dir=output_dir,
        high_quality=True,  # 使用高质量渲染
        image_size=(600, 600)  # 图像大小
    )

    if result:
        # 打印生成摘要
        print_image_generation_summary(result)

        # 验证生成的图像
        verify_generated_images(output_dir)

        # 可选：创建网格图像展示
        print(f"\n创建样本网格图像...")
        generate_image_grid(output_dir, grid_size=(5, 4), output_file="sample_grid.png")

        print(f"\n图像生成完成！")
        print(f"图像文件已保存到: {output_dir}")