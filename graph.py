from graphviz import Digraph
import svgling
import cairosvg
from nltk.tree import Tree
import os

from utils import make_save_dir

# disable nltk image processing.
svgling.disable_nltk_png()


def tree2svg(t):
    """
    converts any nltk tree object to a svg

    Parameters
    ----------
        t: nltk.Tree
           文章の単語のツリー構造

    Returns
    -------

    """

    img = svgling.draw_tree(t, average_glyph_width=1.5)
    svg_data = img.get_svg()
    return svg_data


def generate_images(model_dir, tree_list=[], graph_num=10):
    """
    resultの内容をグラフイメージに変換する。
    変換結果は、model_dir/images 以下に出力される。

    Parameters
    ----------
    model_dir: str
        BERT modelデータの格納されているディレクトリ

    tree_list: list
        画像に変換するTree構造の文字列表現のリスト

    graph_num: int
        graph画像を生成する個数。

    Returns
    -------

    """
    # read from a string and parse the tree using nltk
    images_dir = os.path.join(model_dir, 'images')
    make_save_dir(images_dir)
    for i, line in enumerate(tree_list):
        t = Tree.fromstring(line)
        # convert tree to svg
        sv = tree2svg(t)
        # write the svg as an image
        file = os.path.join(images_dir, f'image_{i}.png')
        cairosvg.svg2png(sv.tostring(), write_to=file, background_color="white")
        print(f'generated {file}.')
        if (i+1) >= graph_num:
            break

