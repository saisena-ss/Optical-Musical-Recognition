# Imports
from PIL import Image, ImageFilter, ImageChops, ImageOps,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import numpy as np
import sys

def hough_transform(image):
    # Convert the image to grayscale
    gray = ImageOps.grayscale(image)
    # Apply Gaussian smoothing
    smoothed = gray.filter(ImageFilter.GaussianBlur(radius=1))
    # Invert the image
    inverted = ImageOps.invert(smoothed)
    edge_image = inverted
    edge_image = edge_image.filter(ImageFilter.MinFilter(3))
    #convert image to array
    edge_array = np.array(edge_image)

    #range of theta, rho and parallel distance
    thetas = np.deg2rad(np.arange(90, 91, 2))#np.deg2rad(np.arange(0, 180, 2))
    height,width = edge_array.shape
    diag_len = height
    rhos = np.arange(0, diag_len, 2)
    spacing = np.arange(0,int(height/10),2)

    accumulator = np.zeros((len(rhos),len(thetas),len(spacing)))

    mod_edge = np.where(edge_array>10,edge_array,0)
    x_idxs = np.nonzero(mod_edge)[1]
    y_idxs = np.nonzero(mod_edge)[0]

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    xcosthetas = np.dot(x_idxs.reshape((-1,1)), cos_theta.reshape((1,-1)))
    ysinthetas = np.dot(y_idxs.reshape((-1,1)), sin_theta.reshape((1,-1)))

    rhosmat = xcosthetas + ysinthetas
    rhosmat = np.ceil(rhosmat/2)*2

#iterate through range of thetas
    for t_idx in range(len(thetas)):
        #iterate through rhos
        for r_idx in range(len(rhos)):
            #iterate through parallel distance
            for s_idx in range(1,len(spacing)):
                #get calculated rho values from rho matrix for the given theta
                rho_dash = rhosmat[:,t_idx]
                #calculate difference between calculated rho and the r_idx
                difference = rho_dash- rhos[r_idx]
                #check if the difference is equal to 1,2,3,4 times spacing and count number of pixels
                # zero_spacing = np.sum(difference==0)
                # one_spacing = np.sum(difference==spacing[s_idx])#[0].shape[0]

                # two_spacing = np.sum(difference==2*spacing[s_idx])#[0].shape[0]
                # three_spacing = np.sum(difference==3*spacing[s_idx])#[0].shape[0]
                # four_spacing = np.sum(difference==4*spacing[s_idx])#[0].shape[0]
                
                # total_votes =  one_spacing+two_spacing+three_spacing+four_spacing +zero_spacing
                temp_spac = [0,spacing[s_idx],2*spacing[s_idx],3*spacing[s_idx],4*spacing[s_idx]]

                total_votes = np.sum(np.isin(difference,temp_spac))#one_spacing+two_spacing+three_spacing+four_spacing+zero_spacing
                
                
                accumulator[r_idx,t_idx,s_idx]+=total_votes

    #get index which contains maximum votes
    max_rho_idx, max_theta_idx, max_spac_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)

    # row,col =
    mr_idx,mt_idx,ms_idx = np.unravel_index(np.argsort(accumulator.ravel()),accumulator.shape)

    accumulator_sub = accumulator[:,max_theta_idx,max_spac_idx]
    mask = accumulator_sub >= 0.7*np.max(accumulator_sub)
    sort_idx = accumulator_sub.argsort()
    final_rhos_idx = sort_idx[mask[sort_idx]]

    #non maximal suppression for hough lines
    final_rhos_idx_nms= []
    for idx in range(len(final_rhos_idx)-1,-1,-1):
        curr_val = final_rhos_idx[idx]
        min_val = final_rhos_idx[idx] - (max_spac_idx*5)
        max_val = final_rhos_idx[idx] + (max_spac_idx*5)
        temp_list = [x for x in range(min_val,max_val+1)]
        is_absent = True
        for ele in temp_list:
            if ele in final_rhos_idx_nms:
                is_absent = False
                break

        if is_absent:
            final_rhos_idx_nms.append(final_rhos_idx[idx])
        
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    max_line_length = 10000

    #theta of first line
    first_line_theta = thetas[max_theta_idx]

    #distance between parallel lines
    dist_prll = spacing[max_spac_idx]

    # for fin_rho in final_rhos_idx_nms:
    #     first_line_rho = rhos[fin_rho]
    #     for line in range(5):
    #         cos_th = np.cos(first_line_theta)
    #         sin_th = np.sin(first_line_theta)
            
    #         x0 = cos_th * (first_line_rho + (line*dist_prll) )
    #         y0 = sin_th * (first_line_rho + (line*dist_prll) )
    #         #first point
    #         x1 = int(x0 + max_line_length * (-sin_th))
    #         y1 = int(y0 + max_line_length * cos_th)
    #         #second point
    #         x2 = int(x0 - max_line_length * (-sin_th))
    #         y2 = int(y0 - max_line_length * cos_th)
    #         # print((x1, y1, x2, y2))
    #         draw.line((x1, y1, x2, y2), fill=(255,0,0), width=1)

    return rhos[final_rhos_idx_nms],dist_prll


def cross_correlation(img, template, threshold=0.7):
    m, n = template.shape[:2]
    img_h, img_w = img.shape[:2]
    result = np.zeros((img_h - m + 1, img_w - n + 1))
    # normalize template
    template = (template - np.mean(template)) / np.std(template)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            roi = img[i:i+m, j:j+n]
            # normalize roi
            roi = (roi - np.mean(roi)) / np.std(roi)
            # compute dot product
            dot_product = (roi * template).sum()
            # compute similarity measure
            similarity = dot_product
            result[i, j] = similarity
    # find locations with similarity above threshold
    locs = np.where(result >= threshold*np.nanmax(result))
    # filter the array to only include values greater than certain threshold
    filtered_arr = result[result > threshold*np.nanmax(result)]
    # get the indices of the filtered values in the original array
    filtered_indices = np.where(result > threshold*np.nanmax(result))
    # get the sorted indices of the filtered values based on their values
    sorted_filtered_indices = np.argsort(filtered_arr)
    # sort the row and column indices in the tuple based on the sorted indices of the filtered values
    sorted_filtered_indices_tuple = (filtered_indices[0][sorted_filtered_indices], filtered_indices[1][sorted_filtered_indices])
    locs = list(zip(sorted_filtered_indices_tuple[1], sorted_filtered_indices_tuple[0]))
    return result,locs

def non_max_suppression(locs,dist_prll,templ_arr):
  nms_list = []
  for loc_idx in range(len(locs)-1,-1,-1):
      top_left = locs[loc_idx]
      is_absent = True
      x = top_left
      temp_list = []
      for i in range(-2*dist_prll,2*dist_prll+1):
          temp_list.append((x[0]+i,x[1]))
          temp_list.append((x[0],x[1]+i))
          temp_list.append((x[0]+i,x[1]+i))
          temp_list.append((x[0]+i,x[1]-i))
          temp_list.append((x[0]-i,x[1]+i))

          for j in range(-2*dist_prll,2*dist_prll+1):
              temp_list.append((x[0]+i,x[1]+j))
              temp_list.append((x[0]-i,x[1]-j))
      for ele in temp_list:
          if ele in nms_list:
              is_absent=False
              break
      if is_absent:
          nms_list.append(top_left)
  return nms_list


def detect_notes(image,line_pos,dist_prll):
    # load image and template
    img = image.convert('L').filter(ImageFilter.MaxFilter(5))
    template = Image.open('template1.png').convert('L')

    img_arr = np.array(img)
    templ_arr = np.array(template.resize((template.width,dist_prll+4)))

    # compute cross-correlation for notes
    result,locs = cross_correlation(img_arr, templ_arr,threshold = 0.7)

    #compute correlation for quarter and eight rest
    template2 = Image.open('template2.png').convert('L')
    templ2_arr = np.array(template2) #.resize((template2.width,2*dist_prll)))
    result2,locs2 = cross_correlation(np.array(image.convert('L')), templ2_arr,threshold=0.9)

    template3 = Image.open('template3.png').convert('L')
    templ3_arr = np.array(template3) #.resize((template2.width,2*dist_prll)))
    result3,locs3 = cross_correlation(np.array(image.convert('L')), templ3_arr,threshold=0.9)

    bass_temp = Image.open('bass_template.png').convert('L')
    bass_result,bass_locs = cross_correlation(np.array(image.convert('L')), np.array(bass_temp),threshold=0.9)

    treble_temp = Image.open('treble_template.png').convert('L')
    treble_result,treble_locs = cross_correlation(np.array(image.convert('L')), np.array(treble_temp),threshold=0.9)

    #doing non-maximal suppression for notes
    nms_list = []
    locs = locs[::-1]
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    for loc in locs:
        top_left = loc
        #generate +4 to left, right, top and bottom
        is_absent = True
        x = loc
        temp_list = []
        for i in range(-dist_prll,dist_prll+1):
            temp_list.append((x[0]+i,x[1]))
            temp_list.append((x[0],x[1]+i))
            temp_list.append((x[0]+i,x[1]+i))
            temp_list.append((x[0]+i,x[1]-i))
            temp_list.append((x[0]-i,x[1]+i))

            for j in range(-dist_prll,dist_prll+1):
                temp_list.append((x[0]+i,x[1]+j))
                temp_list.append((x[0]-i,x[1]-j))
        for ele in temp_list:
            if ele in nms_list:
                is_absent=False
                break
        if is_absent:
            bottom_right = (top_left[0] + templ_arr.shape[1], top_left[1] + templ_arr.shape[0])
            draw.rectangle((top_left, bottom_right), outline='red')
            nms_list.append(loc)
    
    #non maximal suppression for quarter and eight rest
    quarter_rest = non_max_suppression(locs2,template2.height,templ2_arr)
    eight_rest = non_max_suppression(locs3,template2.height,templ3_arr)

    bass_nms_locs = non_max_suppression(bass_locs,bass_temp.height,np.array(bass_temp))
    bass_pos = [x[1] for x in bass_nms_locs]

    treble_nms_locs = non_max_suppression(treble_locs,treble_temp.height,np.array(treble_temp))
    treble_pos = [x[1] for x in treble_nms_locs]

    bass_range = []
    treble_range = []
    for pos in bass_pos:
        bass_range = bass_range + [x for x in range(pos-dist_prll,pos+bass_temp.height +1)]

    for pos in treble_pos:
        treble_range = treble_range + [x for x in range(pos-dist_prll,pos+treble_temp.height+1)]

    final_pos_sorted = line_pos
    final_pos_sorted.sort()
    fnt = ImageFont.truetype("pillowfont.ttf",size=20)


    detected = open("detected.txt","w")

    middle = (dist_prll//2)
    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    for lo in nms_list:
        top_left = lo
        bottom_right = (top_left[0] + templ_arr.shape[1], top_left[1] + templ_arr.shape[0])
        draw.rectangle((top_left[0],top_left[1], bottom_right[0],bottom_right[1]), outline='red')
                
    for first_pos in final_pos_sorted:
        if first_pos in treble_range:
            notes_locs = []
            for loc in nms_list:
                if first_pos - (3*dist_prll) <=loc[1]<= first_pos + (7*dist_prll):
                    notes_locs.append(loc)
            for note_loc in notes_locs:
            #A
                if (first_pos+(2*dist_prll)-2 <= note_loc[1] <= first_pos+(2*dist_prll)+2) or (first_pos-dist_prll-2 <= note_loc[1]+middle <=first_pos-dist_prll+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note A high \n")
                    draw.text((x, y), 'A', font = fnt,fill='red')

                #B
                elif (first_pos+(2*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(2*dist_prll)+2) or (first_pos+(5*dist_prll)-2 <= note_loc[1] <=first_pos+(5*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note B high \n")
                    draw.text((x, y), 'B',font = fnt, fill='red')

                #C
                elif (first_pos+(1*dist_prll)-2 <= note_loc[1] <= first_pos+(1*dist_prll)+2) or (first_pos+(5*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(5*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note C high \n")
                    draw.text((x, y), 'C',font = fnt, fill='red')
                
                #D
                elif (first_pos+(1*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(1*dist_prll)+2) or (first_pos+(4*dist_prll)-2 <= note_loc[1] <=first_pos+(4*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note D high \n")
                    draw.text((x, y), 'D',font = fnt, fill='red')

                #E
                elif (first_pos+(0*dist_prll)-2 <= note_loc[1] <= first_pos+(0*dist_prll)+2) or (first_pos+(4*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(4*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note E high \n")
                    draw.text((x, y), 'E',font = fnt, fill='red')
                
                #F
                elif (first_pos+(0*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(0*dist_prll)+2) or (first_pos+(3*dist_prll)-2 <= note_loc[1] <=first_pos+(3*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note F high \n")
                    draw.text((x, y), 'F',font = fnt, fill='red')
                #G
                elif (first_pos+(0*dist_prll)-2 <= note_loc[1]+dist_prll <= first_pos+(0*dist_prll)+2) or (first_pos+(3*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(3*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note G high \n")
                    draw.text((x, y), 'G',font = fnt, fill='red')
        else:
            notes_bass_locs = []
            for loc in nms_list:
                if first_pos - (3*dist_prll) <=loc[1]<= first_pos + (7*dist_prll):
                    notes_bass_locs.append(loc)
            for note_loc in notes_bass_locs:
            #A
                if (first_pos+(3*dist_prll)-2 <= note_loc[1] <= first_pos+(3*dist_prll)+2) or (first_pos-2 <= note_loc[1]+middle <=first_pos+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note A high \n")
                    draw.text((x, y), 'A',font = fnt, fill='red')

                #B
                elif (first_pos+(3*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(3*dist_prll)+2) or (first_pos+(0*dist_prll)-2 <= note_loc[1]+dist_prll <=first_pos+(0*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note B high \n")
                    draw.text((x, y), 'B',font = fnt, fill='red')

                #C
                elif (first_pos+(2*dist_prll)-2 <= note_loc[1] <= first_pos+(2*dist_prll)+2) or (first_pos-(1*dist_prll)-2 <= note_loc[1]+middle <=first_pos-(1*dist_prll)+2) or (first_pos+(6*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(6*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note C high \n")
                    draw.text((x, y), 'C', font = fnt,fill='red')
                
                #D
                elif (first_pos+(2*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(2*dist_prll)+2)  or (first_pos+(5*dist_prll)-2 <= note_loc[1] <=first_pos+(5*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note D high \n")
                    draw.text((x, y), 'D',font = fnt, fill='red')

                #E
                elif (first_pos+(1*dist_prll)-2 <= note_loc[1] <= first_pos+(1*dist_prll)+2) or (first_pos+(5*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(5*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note E high \n")
                    draw.text((x, y), 'E',font = fnt, fill='red')
                
                #F
                elif (first_pos+(4*dist_prll)-2 <= note_loc[1] <= first_pos+(4*dist_prll)+2) or (first_pos+(1*dist_prll)-2 <= note_loc[1]+middle <=first_pos+(1*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note F high \n")
                    draw.text((x, y), 'F',font = fnt, fill='red')
                
                #G
                elif (first_pos+(4*dist_prll)-2 <= note_loc[1]+middle <= first_pos+(4*dist_prll)+2) or (first_pos+(0*dist_prll)-2 <= note_loc[1] <=first_pos+(0*dist_prll)+2):
                    # define the position and text of the letter
                    x, y = note_loc[0]+20, note_loc[1]
                    # draw the letter
                    detected.write(f"{y} {x} {dist_prll} {template.width} filled_note G high \n")
                    draw.text((x, y), 'G',font = fnt, fill='red')
    for quart_loc in quarter_rest:
        top_left = quart_loc
        bottom_right = (top_left[0] + templ2_arr.shape[1], top_left[1] + templ2_arr.shape[0])
        detected.write(f"{top_left[1]} {top_left[0]} {int(2.4*dist_prll)} {template2.width} quarter_rest _ high \n")
        draw.rectangle((top_left[0],top_left[1], bottom_right[0],bottom_right[1]), outline='green')

    for eight_loc in eight_rest:
        top_left = eight_loc
        bottom_right = (top_left[0] + templ3_arr.shape[1], top_left[1] + templ3_arr.shape[0])
        detected.write(f"{top_left[1]} {top_left[0]} {2*dist_prll} {template3.width} eighth_rest _ high \n")
        draw.rectangle((top_left[0],top_left[1], bottom_right[0],bottom_right[1]), outline='blue')
  
    detected.close()
    return img_rgb


if __name__=='__main__':
    if(len(sys.argv) < 2):
        raise Exception("error: please give an input image name as a parameter, like this: \n"
                     "python3 omr.py music.jpg")

     # Load an image 
    image = Image.open(sys.argv[1])
    # Check its width, height, and number of color channels
    print("Image is %s pixels wide." % image.width)
    print("Image is %s pixels high." % image.height)
    print("Image mode is %s." % image.mode)

    line_pos,dist_prll = hough_transform(image)    
    img_rg = detect_notes(image,line_pos,dist_prll)
    img_rg.save('detected.png')
    print('Code Run Completed')
    # staff_img
