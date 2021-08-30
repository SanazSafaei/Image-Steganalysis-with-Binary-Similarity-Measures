import glob
import Measures

original = glob.glob('original/*.*')
original_copy = original.copy()
stego = glob.glob('stego/*.*')
noise = glob.glob('noise/*.*')
############################################

for i in range(0, 5):
    counter = 0
    svm_input_data = []
    svm_input_label = []
    ##########################################
    svm_test_data = []
    svm_test_label = []
    test_name=[]
    image_numb=0
    flag = 0
    for stego_idx in range(0, len(stego)):

        stego_name = stego[stego_idx]
        print("___________", stego_name, "___________")
        original_name = "original/" + stego_name[14:]
        if original_name not in original_copy:
            original_name = "original/" + stego_name[15:]
        try:
            original.remove(original_name)
        except:
            pass
        print("___________", original_name, "___________")
        measures_pvd = Measures.Measures(original_name, stego_name)
        if i * 40 <= stego_idx < (i + 1) * 40:
            svm_test_data.append(measures_pvd.get_metrics())
            svm_test_label.append("stego")
            test_name.append(str(image_numb)+"_"+stego_name)
            image_numb += 1
        else:
            svm_input_data.append(measures_pvd.get_metrics())
            svm_input_label.append("stego")
    print("===>", len(svm_input_label) , len(original))
    for noise_idx in range(0, len(noise)):

        noise_name = noise[noise_idx]
        print("___________", noise_name, "___________")
        original_name = "original/" + noise_name[6:]
        # if original_name not in original_copy:
        #     original_name = "original/" + noise_name[15:]
        print("___________", original_name, "___________")
        measures_pvd = Measures.Measures(original_name, noise_name)
        if i * 12 <= noise_idx < (i + 1) * 12:
            svm_test_data.append(measures_pvd.get_metrics())
            svm_test_label.append("original")
            test_name.append(str(image_numb) + "_" + noise_name)
            image_numb += 1
        else:
            svm_input_data.append(measures_pvd.get_metrics())
            svm_input_label.append("original")
        try:
            original.remove(original_name)
        except:
            pass
    print("===>", len(svm_input_label), len(original))
    for original_idx in range(0, len(original)):

        original_name = original[original_idx]
        # if original_name in original:
        # original_name = noise_name[15:0]
        # if original_name not in original:
        #     original_name=noise_name[7:0]
        measures_pvd = Measures.Measures(original_name, original_name)
        if i * 21 <= original_idx < (i + 1) * 21:
            svm_test_data.append(measures_pvd.get_metrics())
            svm_test_label.append("original")
            test_name.append(str(image_numb) + "_" + original_name)
            image_numb += 1
        else:
            svm_input_data.append(measures_pvd.get_metrics())
            svm_input_label.append("original")
    print("------------>",len(svm_input_data))
    with open('features/train_input' + str(i) + '.txt', 'w') as filehandle:
        for listitem in svm_input_data:
            filehandle.write('%s|' % listitem)

    with open('features/train_label' + str(i) + '.txt', 'w') as filehandle:
        for listitem in svm_input_label:
            filehandle.write('%s|' % listitem)

    with open('features/test_input' + str(i) + '.txt', 'w') as filehandle:
        for listitem in svm_test_data:
            filehandle.write('%s|' % listitem)

    with open('features/test_label' + str(i) + '.txt', 'w') as filehandle:
        for listitem in svm_test_label:
            filehandle.write('%s|' % listitem)

    with open('features/test_name' + str(i) + '.txt', 'w') as filehandle:
        for listitem in test_name:
            filehandle.write('%s\n' % listitem)


