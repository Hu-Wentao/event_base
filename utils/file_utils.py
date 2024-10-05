import os


def resource_path(file, res_path: str):
    """
    获取相对 .py文件的路径, 用于读取资源文件
    :param file: 固定传入对应.py的 __file__ 动态变量, 代表正在执行的.py文件的位置.(不一定是入口main.py的位置)
    :param res_path: 不带 ./ 或 / 的相对file(.py文件)的路径
    :return:
    """
    current_path = os.path.dirname(os.path.abspath(file))
    # 获取图片的路径
    path = os.path.join(current_path, res_path)
    return path
