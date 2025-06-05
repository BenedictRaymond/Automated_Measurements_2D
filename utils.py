def crop_image(length, width, img):
    img = img[length:1080 - length, width:1920 - width]
    return img
