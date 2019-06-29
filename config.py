
# class_value = [1,5,1,1,1,1, 1, 1,1, 1,1, 1, 1,1,1, 1, 1, 1,1 ,1, 1, 1,1, 1, 1,1, 1,               
#               1, 1, 1,1, 1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1, 1,1, 1, 1,1, 1, 1,
#                1, 1, 1,1, 1, 1,1,
# ]
# class_name = ['Femal',
#               'age',
#               'hair',
#               'hat',
#               'ub-Shirt','ub-Sweater','ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp','ub-Tight', 'ub-ShortSleeve', 'ub-Others',
#               'ub-ColorBlack','ub-ColorWhite', 'ub-ColorGray', 'up-ColorRed', 'ub-ColorGreen','ub-ColorBlue' ,'ub-ColorSilver', 'ub-ColorYellow', 'ub-ColorBrown',
#               'ub-ColorPurple', 'ub-ColorPink', 'ub-ColorOrange','ub-ColorMixture', 'ub-ColorOther',               
#               'lb-LongTrousers', 'lb-Shorts','lb-Skirt', 'lb-ShortSkirt', 'lb-LongSkirt', 'lb-Dress', 'lb-Jeans','lb-TightTrousers', 
#               'lb-ColorBlack', 'lb-ColorWhite', 'lb-ColorGray','lb-ColorRed', 'lb-ColorGreen', 'lb-ColorBlue', 'lb-ColorSilver',
#               'lb-ColorYellow', 'lb-ColorBrown', 'lb-ColorPurple', 'lb-ColorPink','lb-ColorOrange', 'lb-ColorMixture', 'lb-ColorOther',
#               'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag','attachment-WaistBag', 'attachment-Box', 'attachment-PlasticBag',
#               'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Baby','attachment-Other',
#             ]



# class_value = [ 1,
#                 5,
#                 1,
#                 1,
#                 1,1,1,1,1,1,1,1,1,1,
#                 1,1,1,1,1,1,1,1,1,
#                 1,1,1,1,1,
#                 1,1,1,1,1,1,1,1,
#                 1,1,1,1,1,1,1,
#                 1,1,1,1,1,1,1,
#                 1,1,1,1,1,1,
#                 1,1,1,1,
# ]
# class_name = ['Femal',
#               'age',
#               'hair',
#               'hat',
#               'ub-Shirt','ub-Sweater','ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp','ub-Tight', 'ub-ShortSleeve', 'ub-Others',
#               'ub-ColorBlack','ub-ColorWhite', 'ub-ColorGray', 'up-ColorRed', 'ub-ColorGreen','ub-ColorBlue' ,'ub-ColorSilver', 'ub-ColorYellow', 'ub-ColorBrown',
#               'ub-ColorPurple', 'ub-ColorPink', 'ub-ColorOrange','ub-ColorMixture', 'ub-ColorOther',               
#               'lb-LongTrousers', 'lb-Shorts','lb-Skirt', 'lb-ShortSkirt', 'lb-LongSkirt', 'lb-Dress', 'lb-Jeans','lb-TightTrousers', 
#               'lb-ColorBlack', 'lb-ColorWhite', 'lb-ColorGray','lb-ColorRed', 'lb-ColorGreen', 'lb-ColorBlue', 'lb-ColorSilver',
#               'lb-ColorYellow', 'lb-ColorBrown', 'lb-ColorPurple', 'lb-ColorPink','lb-ColorOrange', 'lb-ColorMixture', 'lb-ColorOther',
#               'attachment-Backpack', 'attachment-ShoulderBag', 'attachment-HandBag','attachment-WaistBag', 'attachment-Box', 'attachment-PlasticBag',
#               'attachment-PaperBag', 'attachment-HandTrunk', 'attachment-Baby','attachment-Other',
#             ]



#合并了一些bag标签
class_value = [ 1,
                5,
                1,
                1,
                10,
                14,
                8,
                14,
                1,1,1,1,1,
] 
class_name = ['Femal',
              'age',
              'hair',
              'hat',
              'ub-type',
              'ub-color',          
              'lb-type', 
              'lb-color', 
              'attachment-Bag',
              'attachment-HandBag',
              'attachment-Box', 
              'attachment-HandTrunk', 
              'attachment-Baby',
            ]

# attachment-bag:4706
# attachment-HandBag:3229
# attachment-Box:1911
# attachment-HandTrunk:1244
# attachment-Baby:167


subclass = {
    'Femal':{'female','male'},
    'age':{'AgeLess16', 'Age17-30', 'Age31-45', 'Age46-60','AgeBiger60'},
    'hair':{'long','short'},
    'hat':{'hat','nohat'},
    'up_clothestype':{'ub-Shirt','ub-Sweater','ub-Vest', 'ub-TShirt', 'ub-Cotton', 'ub-Jacket', 'ub-SuitUp','ub-Tight', 'ub-ShortSleeve', 'ub-Others',},
    'up_clothescolor':{'ub-ColorBlack','ub-ColorWhite', 'ub-ColorGray', 'up-ColorRed', 'ub-ColorGreen','ub-ColorBlue' ,'ub-ColorSilver', 'ub-ColorYellow', 'ub-ColorBrown',
                        'ub-ColorPurple', 'ub-ColorPink', 'ub-ColorOrange','ub-ColorMixture', 'ub-ColorOther'},
    'lb_clothestype':{'lb-LongTrousers', 'lb-Shorts','lb-Skirt', 'lb-ShortSkirt', 'lb-LongSkirt', 'lb-Dress', 'lb-Jeans','lb-TightTrousers'},
    'lb_clothescolor':{'lb-ColorBlack', 'lb-ColorWhite', 'lb-ColorGray','lb-ColorRed', 'lb-ColorGreen', 'lb-ColorBlue', 'lb-ColorSilver',
                        'lb-ColorYellow', 'lb-ColorBrown', 'lb-ColorPurple', 'lb-ColorPink','lb-ColorOrange', 'lb-ColorMixture', 'lb-ColorOther'},
    'attachment-bag':{'attachment-Backpack','no_attachment-Backpack'},
    'attachment-HandBag':{'attachment-hangbag','no_attachment-hangbag'},
    'attachment-Box':{'attachment-Box','no_attachment-Box'},
    'attachment-HandTrunk':{'attachment-HandTrunk','no_attachment-HandTrunk'},
    'attachment-Baby':{'attachment-Baby','no_attachment-Baby'},
}

origin_label = ['Femal:1'
             'AgeLess16:1' 'Age17-30:1' 'Age31-45:1' 'Age46-60:1' 'AgeBiger60:1' 
             'hs-LongHair:1'
            'hs-Hat:1'
            'ub-Shirt:1' 'ub-Sweater:1' 'ub-Vest:1' 'ub-TShirt:1' 'ub-Cotton:1' 'ub-Jacket:1' 'ub-SuitUp:1' 'ub-Tight:1' 'ub-ShortSleeve:1' 'ub-Others:1' 
            'ub-ColorBlack:1' 'ub-ColorWhite:1' 'ub-ColorGray:1' 'up-ColorRed:1' 'ub-ColorGreen:1' 'ub-ColorBlue:1' 'ub-ColorSilver:1' 'ub-ColorYellow:1' 'ub-ColorBrown:1'
            'ub-ColorPurple:1' 'ub-ColorPink:1' 'ub-ColorOrange:1' 'ub-ColorMixture:1' 'ub-ColorOther:1' 'lb-LongTrousers:1' 'lb-Shorts:1'
            'lb-Skirt:1' 'lb-ShortSkirt:1' 'lb-LongSkirt:1' 'lb-Dress:1' 'lb-Jeans:1' 
 'lb-TightTrousers:1' 'lb-ColorBlack:1' 'lb-ColorWhite:1' 'lb-ColorGray:1'
 'lb-ColorRed:1' 'lb-ColorGreen:1' 'lb-ColorBlue:1' 'lb-ColorSilver:1'
 'lb-ColorYellow:1' 'lb-ColorBrown:1' 'lb-ColorPurple:1' 'lb-ColorPink:1'
 'lb-ColorOrange:1' 'lb-ColorMixture:1' 'lb-ColorOther:1'
 'attachment-Backpack:1' 'attachment-ShoulderBag:1' 'attachment-HandBag:1'
 'attachment-WaistBag:1' 'attachment-Box:1' 'attachment-PlasticBag:1'
 'attachment-PaperBag:1' 'attachment-HandTrunk:1' 'attachment-Baby:1'
 'attachment-Other:1']