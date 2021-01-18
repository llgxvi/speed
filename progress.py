def progress(current, total, barLength = 20):
    percent = float(current) / total * 100
    arrow   = '=' * int(percent / 100 * barLength - 1) + '>'
    spaces  = '.' * (barLength - len(arrow))

    if percent == 100:
        arrow = arrow[:-1] + '='
        print('Progress: [=%s%s] %.0f%%   ' % (arrow, spaces, percent), end='\n')
    else:
        print('Progress: [=%s%s] %.2f%%' % (arrow, spaces, percent), end='\r')
