import SessionUtils from '../SessionUtils'

const state = {
    FileName: SessionUtils.getValue('FileName'),
    Date: SessionUtils.getValue('Date'),
    Status: SessionUtils.getValue('Status'),
    FileId: SessionUtils.getValue('FileId'),
    NeedEmail: SessionUtils.getValue('NeedEmail') || false,
    TestResult: SessionUtils.getValue('TestResult'),
  }

const mutations = {
    UPDATE_FileName: (state, FileName) => {
        state.FileName = FileName
        SessionUtils.setValue('FileName', FileName)
    },
    UPDATE_Date: (state, Date) => {
        state.Date = Date
        SessionUtils.setValue('Date', Date)
    },
    UPDATE_Status: (state, Status) => {
        state.Status = Status    
        SessionUtils.setValue('Status', Status)
    },
    UPDATE_FileId: (state, FileId) => {
        state.FileId = FileId    
        SessionUtils.setValue('FileId', FileId)
    },
    UPDATE_TestResult: (state, TestResult) => {
        state.TestResult = TestResult    
        SessionUtils.setValue('TestResult', TestResult)
    },
}
  
export const submit = {
    namespaced: true,
    state,
    mutations
}