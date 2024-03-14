import SessionUtils from '../SessionUtils'

const state = {
    // tableData: SessionUtils.getValue('tableData'),
    tableData: Array,
    // GetRecord: SessionUtils.getValue('GetRecord'),
    GetRecord: false,
}

const mutations = {
    UPDATE_tableData: (state, tableData) => {
        state.tableData = tableData
        // SessionUtils.setValue('tableData', tableData)
    },
    UPDATE_Record: (state, record) => {
        state.tableData.unshift(record)
        // SessionUtils.setValue('tableData', state.tableData)
    },
    UPDATE_Status: (state, idx, Status) => {
        state.tableData[idx].Status = Status
        // SessionUtils.setValueDeep('tableData', idx, Status)
    },
    UPDATE_GetRecord: (state, GetRecord) => {
        state.GetRecord = GetRecord
        // SessionUtils.setValue('GetRecord', GetRecord)
    },
}
  
export const record = {
    namespaced: true,
    state,
    mutations
}