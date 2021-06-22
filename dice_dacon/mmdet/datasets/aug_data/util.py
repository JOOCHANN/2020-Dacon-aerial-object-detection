import os
def img_list_loader(path, extension = 'pickle' ):
    """
    :param path: 이미지를 불러올 디렉토리 명입니다.
    :param extension: 선택할 확장자 입니다.
    :return: 파일 경로가 있는 imgs_list를 불러옵니다.
    """
    imgs_list = os.listdir(path)
    imgs_list = sorted(imgs_list)
    result = \
        [os.path.join(path, name) for name in imgs_list if name.split(".")[1] == extension]
    return result
