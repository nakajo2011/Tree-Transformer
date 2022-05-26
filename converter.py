import os
import re
import pandas as pd
from datetime import datetime

from utils import make_save_dir


class SortIdConverter(object):
    """
    model_dir/result/tree.txtのsortid部分を番組のメタ情報に変換するためのクラス
    """

    def __init__(
            self,
            model_dir=None,
            tv_meta_data_file=None,
            special_tokens = {'[CLS]', '[SEP]', '[MASK]'}
    ) -> None:
        """
        Parameters
        ----------
        model_dir : str
            model_dirのパス

        tv_meta_data_file : str
            番組のメタ情報のcsvファイルのパス
        """

        tree_file = os.path.join(model_dir, "result/tree.txt")
        assert(model_dir is not None), 'model_dir not specify.'
        assert(os.path.isfile(tree_file), f'{tree_file} is not exists.')

        self.converted_dir = os.path.join(model_dir, 'converted')
        make_save_dir(self.converted_dir)  # 変換後のファイルを格納するディレクトリ

        with open(tree_file) as f:
            self.tree_txt = f.readlines()

        self.convertable = os.path.exists(tv_meta_data_file)
        # 番組メタ情報をロード
        self.meta_df = []
        if self.convertable:
            self.meta_df = pd.read_csv(tv_meta_data_file)

        self.special_tokens = special_tokens

        self.channel_table = {
            'ＮＨＫ総合': 'N',
            'ＴＢＳ': 'T',
            'テレビ朝日': 'A',
            'テレビ東京': 'B',
            'フジテレビ': 'F',
            '日本テレビ': 'H'
        }
        self.tvname = {}  # 番組情報をindex値に変換する時に使う辞書
        self.convert_infos = []  # sortidとショート番組情報と番組の詳細情報とのマッピング情報を保管するリスト

    def convert(self) -> dict:
        """
        analysis.jsonのsortid部分を番組情報に変換する
        元々のsortidと番組情報とショート番組情報の対応表を別途ファイルに出力する

        Returns
        -------
            tree構造文字列の中のsortidを番組情報に変換した文字列
        """

        if self.convertable:
            result = []
            for i, stree in enumerate(self.tree_txt):
                self.tvname = {}
                self.convert_infos = []
                result.append(re.sub(r'162[0-9]{8}', self._convert_sortid2meta_info, stree))

                info_file = os.path.join(self.converted_dir, f'{i}_convert_info.txt')
                with open(info_file, mode='w') as w:
                    w.write('\n'.join(self.convert_infos))

        return self.tree_txt

    def _convert_sortid2meta_info(self, sortid=None):
        """
        sortidを番組の放送時間、放送局、番組名、サブタイトルに変換して返す。
        渡されたsortidがspecial token([CLS]、[SEP]、[MASK])の場合はそのまま返す。

        Parameters
        ----------
        sortid: str
            番組のコマを表すid もしくは special token

        metadata_df: DataFrame
            番組のメタ情報

        Returns
        -------
            番組の放送時間、放送局、番組名、サブタイトルを結合した文字列

        """

        print(f'sortid={sortid.group()}')

        sortid = sortid.group()
        if sortid in self.special_tokens:
            return sortid

        target_df = self.meta_df.query(f'sortid == {sortid}')
        channel = target_df['放送局'].values[0]  # 放送局
        name = channel + target_df['番組名'].values[0] + str(target_df['サブタイトル'].values[0])  # idに変換するための番組名
        sdt = target_df['from'].values[0] # 放送時間
        dt = datetime.strptime(sdt, '%Y/%m/%d %H:%M:%S')
        formatted_date = dt.strftime("%m%dT%H%M")

        if not (name in self.tvname):
            self.tvname |= {name: len(self.tvname)}

        short_name = f'{self.channel_table[channel]}{self.tvname[name]}_{formatted_date}'
        data_list = list(target_df[['from', '放送局', '番組名', 'サブタイトル']].values[0])

        print(f'番組名={name}, {self.tvname[name]}')
        convert_info = ' '.join([sortid, short_name, *map(str, data_list)])
        self.convert_infos.append(convert_info)
        return short_name

