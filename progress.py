def progress(current, total, barLength = 20):
    percent = float(current) / total * 100
    arrow   = '-' * int(percent / 100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    if percent == 100:
        print('Progress: [-%s%s] %f%%' % (arrow, spaces, percent), end='\n')
    else:
        print('Progress: [-%s%s] %.2f%%' % (arrow, spaces, percent), end='\r')
