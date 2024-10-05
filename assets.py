import os.path

from utils.file_utils import resource_path


def path_data(sub: str):
    return resource_path(__file__, os.path.join('data', sub))


def db_uri_database(override=None) -> str:
    return path_data(override or 'event_base.db')


if __name__ == '__main__':
    print(db_uri_database('sadf.db'))
    print(db_uri_database())
