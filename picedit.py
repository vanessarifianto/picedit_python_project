import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# C:\Users\62896\Downloads\ps1\test.png
# C:\Users\62896\Downloads\ps1\mini_test.png

def load_image(filename):
    img = mpimg.imread(filename)
    if img.shape[-1] == 4:
        img = img[..., :3]
        print("Converted RGBA image to RGB.")

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        print("Converted image to uint8 format.")

    mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    print(f"Image '{filename}' loaded successfully.")
    return img, mask


def save_image(filename, img):
    mpimg.imsave(filename, img)
    print(f"Image saved as '{filename}'.")


def display_image(img, mask=None):
    if mask is None:
        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)
    plt.imshow(img, alpha=0.8)
    plt.imshow(mask, cmap='gray', alpha=0.3)
    plt.axis('off')
    plt.show()

def add_to_history(history, redo_stack, img, mask):
    history.append((img.copy(), mask.copy()))
    redo_stack.clear()

def undo(history, redo_stack):
    if len(history) > 1:
        redo_stack.append(history.pop())
        return history[-1]
    else:
        print("No more actions to undo.")
        return history[-1]

def redo(history, redo_stack):
    if redo_stack:
        state = redo_stack.pop()
        history.append(state)
        return state
    else:
        print("No more actions to redo.")
        return history[-1]

def change_brightness(image, value, mask=None):
    try:
        if type(value) != int or value < -255 or value > 255:
            raise ValueError("Brightness value must be an integer between -255 and 255.")

        if mask is None:
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        new_image = image.copy().astype('int32')
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if mask[r, c] == 1:
                    for ch in range(image.shape[2]):
                        new_value = new_image[r, c, ch] + value
                        new_image[r, c, ch] = np.clip(new_value, 0, 255)

        return new_image.astype('uint8')
    except ValueError as e:
        print(e)
        return image


def change_contrast(image, value, mask=None):
    try:
        if type(value) != int or value < -255 or value > 255:
            raise ValueError("Contrast value must be an integer between -255 and 255.")

        if mask is None:
            mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

        factor = (259 * (value + 255)) / (255 * (259 - value))
        new_image = image.copy().astype('int32')
        for r in range(image.shape[0]):
            for c in range(image.shape[1]):
                if mask[r, c] == 1:
                    for ch in range(image.shape[2]):
                        new_value = factor * (new_image[r, c, ch] - 128) + 128
                        new_image[r, c, ch] = np.clip(new_value, 0, 255)

        return new_image.astype('uint8')
    except ValueError as e:
        print(e)
        return image


def grayscale(image, mask=None):
    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    new_image = image.copy()
    gray_image = np.dot(image[..., :3], [0.3, 0.59, 0.11])
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if mask[r, c] == 1:
                new_image[r, c, :3] = gray_image[r, c]
    return new_image


def blur_effect(image, mask=None):
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125, 0.25, 0.125],
                       [0.0625, 0.125, 0.0625]])

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    blurred_image = image.copy()
    for i in range(3):
        for r in range(1, image.shape[0] - 1):
            for c in range(1, image.shape[1] - 1):
                if mask[r, c] == 1:
                    region = image[r - 1:r + 2, c - 1:c + 2, i]
                    blurred_image[r, c, i] = np.clip(np.sum(region * kernel), 0, 255)
    return blurred_image


def edge_detection(image, mask=None):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    new_image = image.copy()
    for r in range(1, image.shape[0] - 1):
        for c in range(1, image.shape[1] - 1):
            if mask[r, c] == 1:
                for ch in range(3):
                    region = image[r - 1:r + 2, c - 1:c + 2, ch]
                    new_image[r, c, ch] = np.clip(np.sum(region * kernel) + 128, 0, 255)
    return new_image


def embossed(image, mask=None):
    kernel = np.array([[-1, -1, 0],
                       [-1, 0, 1],
                       [0, 1, 1]])

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    embossed_image = image.copy()
    for i in range(3):
        for r in range(1, image.shape[0] - 1):
            for c in range(1, image.shape[1] - 1):
                if mask[r, c] == 1:
                    region = image[r - 1:r + 2, c - 1:c + 2, i]
                    embossed_image[r, c, i] = np.clip(np.sum(region * kernel) + 128, 0, 255)
    return embossed_image


def rectangle_select(image, x, y):
    x1, y1 = x
    x2, y2 = y
    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    new_mask[x1:x2 + 1, y1:y2 + 1] = 1
    print(f"Rectangle selection from top-left ({x1}, {y1}) to bottom-right ({x2}, {y2}) applied.")
    return new_mask


def calculate_color_distance(pixel1, pixel2):
    r_mean = (pixel1[0] + pixel2[0]) / 2
    delta_r = pixel1[0] - pixel2[0]
    delta_g = pixel1[1] - pixel2[1]
    delta_b = pixel1[2] - pixel2[2]
    return np.sqrt((2 + r_mean / 256) * delta_r ** 2 + 4 * delta_g ** 2 + (2 + (255 - r_mean) / 256) * delta_b ** 2)


def magic_wand_select(image, x, threshold):
    x_coord, y_coord = x
    new_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    initial_pixel = image[x_coord, y_coord].astype(np.float32)
    stack = [(x_coord, y_coord)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1] and new_mask[r, c] == 0:
            current_pixel = image[r, c].astype(np.float32)
            if calculate_color_distance(current_pixel, initial_pixel) <= threshold:
                new_mask[r, c] = 1
                stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return new_mask


def apply_warm_tone(image, intensity=0.5, mask=None):
    if not (0 <= intensity <= 1):
        raise ValueError("Intensity must be between 0 and 1.")

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    new_image = image.copy().astype('int32')
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if mask[r, c] == 1:
                new_image[r, c, 0] = np.clip(new_image[r, c, 0] * (1 + 0.2 * intensity), 0, 255)
                new_image[r, c, 1] = np.clip(new_image[r, c, 1] * (1 + 0.1 * intensity), 0, 255)
                new_image[r, c, 2] = np.clip(new_image[r, c, 2] * (1 - 0.1 * intensity), 0, 255)

    return new_image.astype('uint8')


def apply_cool_tone(image, intensity=0.5, mask=None):
    if not (0 <= intensity <= 1):
        raise ValueError("Intensity must be between 0 and 1.")

    if mask is None:
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    new_image = image.copy().astype('int32')
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if mask[r, c] == 1:
                new_image[r, c, 0] = np.clip(new_image[r, c, 0] * (1 - 0.2 * intensity), 0, 255)
                new_image[r, c, 1] = np.clip(new_image[r, c, 1] * (1 + 0.1 * intensity), 0, 255)
                new_image[r, c, 2] = np.clip(new_image[r, c, 2] * (1 + 0.2 * intensity), 0, 255)

    return new_image.astype('uint8')


def menu():
    img = None
    mask = None
    history = []
    redo_stack = []

    while True:
        if img is None:
            print("No image loaded. Please load an image first.")
            print("e - exit")
            print("l - load a picture")
            choice = input("Your choice: ")

            if choice == 'e':
                break
            elif choice == 'l':
                filename = input("Enter the filename (e.g., 'example.png'): ")
                img, mask = load_image(filename)
                history.clear()
                redo_stack.clear()
                history.append((img.copy(), mask.copy()))
            else:
                print("Invalid choice.")
        else:
            print("Options:")
            print("e - exit")
            print("l - load a new picture")
            print("s - save the current picture")
            print("1 - change brightness")
            print("2 - change contrast")
            print("3 - apply grayscale")
            print("4 - apply blur")
            print("5 - edge detection")
            print("6 - embossed effect")
            print("7 - rectangle selection")
            print("8 - magic wand selection")
            print("9 - apply warm tone")
            print("10 - apply cool tone")
            print("z - undo last action")
            print("y - redo undone action")
            choice = input("Your choice: ")

            if choice == 'e':
                break
            elif choice == 'l':
                filename = input("Enter the filename for the new image (e.g., 'example.png'): ")
                img, mask = load_image(filename)
                history.clear()
                redo_stack.clear()
                history.append((img.copy(), mask.copy()))
            elif choice == 's':
                filename = input("Enter the filename to save (e.g., 'output.png'): ")
                save_image(filename, img)
            elif choice == '1':
                value = int(input("Enter brightness adjustment (-255 to 255): "))
                img = change_brightness(img, value, mask)
            elif choice == '2':
                value = int(input("Enter contrast adjustment (-255 to 255): "))
                img = change_contrast(img, value, mask)
            elif choice == '3':
                img = grayscale(img, mask)
            elif choice == '4':
                img = blur_effect(img, mask)
            elif choice == '5':
                img = edge_detection(img, mask)
            elif choice == '6':
                img = embossed(img, mask)
            elif choice == '7':
                x1, y1 = map(int, input("Enter top-left corner coordinates (x1 y1): ").split())
                x2, y2 = map(int, input("Enter bottom-right corner coordinates (x2 y2): ").split())
                mask = rectangle_select(img, (x1, y1), (x2, y2))
            elif choice == '8':
                x, y = map(int, input("Enter starting pixel coordinates (x y): ").split())
                threshold = float(input("Enter the threshold: "))
                mask = magic_wand_select(img, (x, y), threshold)
            elif choice == '9':
                intensity = float(input("Enter intensity for warm tone adjustment (0 to 1): "))
                img = apply_warm_tone(img, intensity, mask)
            elif choice == '10':
                intensity = float(input("Enter intensity for cool tone adjustment (0 to 1): "))
                img = apply_cool_tone(img, intensity, mask)
            elif choice == 'z':
                img, mask = undo(history, redo_stack)
            elif choice == 'y':
                img, mask = redo(history, redo_stack)
            else:
                print("Invalid choice.")

            display_image(img, mask)
            history.append((img.copy(), mask.copy()))


if __name__ == "__main__":
    menu()
