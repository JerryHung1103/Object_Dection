from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import cv2
# print(torch.cuda.is_available())
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
yellow = (0, 255, 255) # in BGR 
font = cv2.FONT_HERSHEY_SIMPLEX
stroke = 2 
# print(model.config.id2label)
# default webcam
stream = cv2.VideoCapture(0)
# stream.set(cv2.CAP_PROP_FRAME_WIDTH, 3000)
# stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 3000)
while(True):
    # Capture frame-by-frame

    (grabbed, frame) = stream.read()

    # convert the image from NumPy array into a PIL image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img)
    inputs = processor(images = image, return_tensors = "pt")
    inputs = inputs.to(device)
    outputs = model(**inputs)
    # print(img.shape)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes = target_sizes, 
        threshold = 0.7)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"        
        )

        # draw the bounding box
        cv2.rectangle(frame, 
                      (int(box[0]), int(box[1])),   # x1, y1
                      (int(box[2]), int(box[3])),   # x2, y2
                      yellow, 
                      stroke)
        
        # display the label
        cv2.putText(frame, 
                    model.config.id2label[label.item()], # label
                    (int(box[0]), int(box[1]-10)),       # x1, y1
                    font, 
                    1, 
                    yellow, 
                    stroke, 
                    cv2.LINE_AA)
 

# #     # Show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF    

    if key == ord("q"):    # Press q to break out of the loop
        break
    del frame, img, image, inputs, outputs, results
    torch.cuda.empty_cache()

# Cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)