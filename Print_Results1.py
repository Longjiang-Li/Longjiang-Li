
def Print_Results_title(filename='',batch_size=0,lr=0,lr_decay='',optimizer_type='', num_classes=0):
    """
    训练文件头
    """
    with open(filename, 'w') as txt:
        str_bs  = 'Batch_Size'
        str_lr  = 'Learning_Rate'
        str_lrd = 'Learning_Rate_Decay'
        str_ot  = 'Optimizer_Type'
        txt.write('{} {} {} {} \n'.format(str_bs, str_lr, str_lrd, str_ot))
        batch_size     = str(batch_size).rjust(len(str_bs))
        lr             = str(lr).rjust(len(str_lr))
        lr_decay       = lr_decay.rjust(len(str_lrd))
        optimizer_type = optimizer_type.rjust(len(str_ot))

        txt.write('{} {} {} {} \n'.format(batch_size, lr, lr_decay, optimizer_type))
        txt.write('\n')
        txt.write('Epoch'.rjust(5))
        txt.write('TN'.rjust(10))
        txt.write('FP'.rjust(10))
        txt.write('FN'.rjust(10))
        txt.write('TP'.rjust(10))
        txt.write('FAR'.rjust(10))
        txt.write('POD'.rjust(10))
        txt.write('Acc'.rjust(10))
        txt.write('TSS'.rjust(10))
        txt.write('CSI'.rjust(10))
        txt.write('Loss'.rjust(10))
        txt.write('\n')

def Print_Results_data(filename='', epoch=0, TN=0, FP=0, FN=0, TP=0, FAR=0, POD=0, ACC=0, TSS=0, CSI=0, epoch_loss=0):
    
    with open(filename, 'a') as txt:
        txt.write('{}'.format(str(epoch).rjust(5)))
        txt.write('{}'.format(str(TN).rjust(10)))
        txt.write('{}'.format(str(FP).rjust(10)))
        txt.write('{}'.format(str(FN).rjust(10)))
        txt.write('{}'.format(str(TP).rjust(10)))
        txt.write('{:10.2f}'.format(100 * FAR))
        txt.write('{:10.2f}'.format(100 * POD))
        txt.write('{:10.2f}'.format(100 * ACC))
        txt.write('{:10.2f}'.format(100 * TSS))
        txt.write('{:10.2f}'.format(100 * CSI))
        txt.write('{:10.2f}\n'.format(epoch_loss))