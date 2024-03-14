<template>
  <div v-if="LoggedIn">
    <el-space wrap>
      <el-card class="box-card">
        <template #header>
          <div class="card-header" style="height: 180px">
            <Upload></Upload>
          </div>
        </template>
        <div class="text item">
          <h1>纠错结果</h1>
          <span>&nbsp</span>
          <el-alert style="width: 1000px;" title="请点击记录跳转到详情页面" type="info" :closable="false"/>
          <span>&nbsp</span>
          <div v-if="GetRecord">
            <el-table :data="tableData" 
            height="500" 
            style="width: 100%"
            @row-click="toDetail" >
              <el-table-column prop="FileId" label="文件ID" width="100px" />
              <el-table-column prop="Date" label="提交日期" width="200px" />
              <el-table-column prop="FileName" label="文件名" min-width="400px"
                  :show-overflow-tooltip="true" width="400px" />
              <el-table-column prop="Status" label="状态" :show-overflow-tooltip="true" width="200px"/>
              <el-table-column fixed="right" label="" width="30px">
                <template #default="scope">
                  <!-- <el-button link type="primary" @click="toDetail(scope.row)"
                    >查看</el-button
                  > -->
                  <el-dropdown trigger="click">
                    <el-button link @click.native.stop="Row = scope.row" type="primary" style="font-size: 18px;">
                      &nbsp&nbsp⋮
                    </el-button>
                    <template #dropdown>
                      <el-dropdown-menu>
                        <el-dropdown-item>
                          <el-button link type="primary" @click.native.stop="MyDownload(scope.row)">
                            下载原文
                          </el-button>
                        </el-dropdown-item>
                        <el-dropdown-item>
                          <el-button link type="primary" @click.native.stop="MyDelete(scope.row)">
                            删除
                          </el-button>
                        </el-dropdown-item>
                        <!-- <el-dropdown-item>
                          <el-button link type="primary" @click.native.stop="MyPrivacy(scope.row)">
                            文档设置
                          </el-button>
                        </el-dropdown-item> -->
                      </el-dropdown-menu>
                    </template>
                  </el-dropdown>
                </template>
              </el-table-column>
            </el-table>
            <el-dialog
              v-model="dialog_delete"
              title="是否确认删除？"
              width="400px"
              destroy-on-close
              align-center
            >
              <div>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-checkbox v-model="whether_delete_data" label="同时在后台数据库删除我的数据" size="large" />
              </div>
              <div>&nbsp</div>
              <div>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="primary" @click="Delete(Row)">删除</el-button>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button link size="small" type="default" @click.prevent="cansel">取消</el-button>
              </div>
            </el-dialog>
            <!-- <el-dialog
              v-model="dialog_privacy"
              title="文档设置"
              width="400px"
              destroy-on-close
              align-center
            >
            隐私设置：
              <el-select v-model="fileprivacy" placeholder="可公开使用">
                <el-option label="可公开使用" value="可公开使用" />
                <el-option label="乐改团队内部使用" value="乐改团队内部使用" />
                <el-option label="不可使用" value="不可使用" />
              </el-select>
              <div>&nbsp</div>
              <div>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="primary" @click="FilePrivacy(Row)">保存</el-button>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button link size="small" type="default" @click.prevent="dialog_privacy = false">取消</el-button>
              </div>
            </el-dialog> -->
          </div>
          <div v-else>
            正在获取提交记录...
          </div>
        </div>
      </el-card>
    </el-space>  
  </div>
  <div v-else>
    <el-space wrap>
      <el-card class="box-card" style="width: 1000px;">
        <template #header>
          <div class="card-header">
            请先登录
          </div>
        </template>
        <div class="text item">
          <h1>提交记录</h1>
        </div>
      </el-card>
    </el-space>  
  </div>
</template>
  
<script lang="ts" setup>
  import router from '@/router';
  import axios from 'axios';
  import { computed, ref, watch } from 'vue'
  import { createLogger, useStore } from 'vuex'
  import Upload from './upload.vue'
  import { ElMessage, ElMessageBox } from 'element-plus'
  import type { Action } from 'element-plus'
  import { stringLiteral } from '@babel/types';

  const store = useStore()
  const LoggedIn = computed(() => store.state.log.LoggedIn || false)
  const GetRecord = computed(() => store.state.record.GetRecord || false)
  const dialog_delete = ref(false)
  const dialog_privacy = ref(false)
  const whether_delete_data = ref(false)
  const fileprivacy = ref("可公开使用")

  let Row = {
    FileId: "0",
  }

  // 后端请求提交记录
  watch(LoggedIn, () => {
    if(LoggedIn.value) {
      let formData = new FormData();
      console.log("请求记录",store.state.log.UserName);
      formData.append("UserName", store.state.log.UserName);
      axios.post('/record', formData).then(res => {
        if(res.data.code != 707){
          store.commit('record/UPDATE_tableData', res.data)
          console.log("请求记录成功",store.state.log.UserName)
          store.commit('record/UPDATE_GetRecord', true)
        }
        else{
          store.commit('log/UPDATE_LOG', false)
          store.commit('log/UPDATE_EXIT', false)
        }
        if (res.data.message) {
          ElMessage(res.data.message)
        }
      })
    }
  });
  
  if(LoggedIn.value) {
    let formData = new FormData();
    console.log("请求记录",store.state.log.UserName);
    formData.append("UserName", store.state.log.UserName);
    axios.post('/record', formData).then(res => {
      if(res.data.code != 707){
        store.commit('record/UPDATE_tableData', res.data)
        console.log("请求记录成功",store.state.log.UserName)
        store.commit('record/UPDATE_GetRecord', true)
      }
      else{
        store.commit('log/UPDATE_LOG', false)
        store.commit('log/UPDATE_EXIT', false)
      }
      if (res.data.message) {
        ElMessage(res.data.message)
      }
    })
  }
  const tableData = computed(() => store.state.record.tableData)

// 后端请求对应的纠错结果
const toDetail = (row:{FileId:string, [propName:string]:any}) => {
  let formData = new FormData();
  formData.append("FileId", row.FileId);
  formData.append("UserName", store.state.log.UserName);
  formData.append("ResultMode", "均衡");
  axios.post('/correct', formData).then(res => {
    if(res.data[4] != 707){
      if(Array.isArray(res.data[2])) {
        store.commit('result/UPDATE_GECRESULT', res.data)
        console.log("纠错参考更新完成")
        console.log(row.FileId)
        store.commit('result/UPDATE_FileId', row.FileId)
        if ((window.getSelection() as Selection).toString()===''){
          router.push({ path: "/detail", query: { FileId: row.FileId } })
        }
      }
      else if(res.data[3]=="纠错完成") {
        ElMessageBox.alert(res.data[3]+"，无错误句子", '提示', {
          confirmButtonText: 'OK',
          callback: (action: Action) => {
            ElMessage({
              type: 'info',
              message: res.data[3]+"，无错误句子",
            })
          },
        })
      }
      else {
        ElMessageBox.alert(res.data[3], '提示', {
          confirmButtonText: 'OK',
          callback: (action: Action) => {
            ElMessage({
              type: 'info',
              message: res.data[3],
            })
          },
        })
      }
    }
    else{
      store.commit('log/UPDATE_LOG', false)
      store.commit('log/UPDATE_EXIT', false)
    }
    if (res.data[3]) {
      ElMessage(res.data[3])
    }
  })
}

const cansel = (row:{FileId:string, [propName:string]:any}) => {
  dialog_delete.value = false
}

const MyDelete = (row:{FileId:string, [propName:string]:any}) => {
  dialog_delete.value = true
  Row = row
}
const MyDownload = (row:{FileId:string, [propName:string]:any}) => {
  let formData = new FormData();
  formData.append("FileId",row.FileId);
  formData.append("UserName",store.state.log.UserName);
  axios.post('/download', formData, { responseType: "blob" }).then(res => {
    const blob = new Blob([res.data])
      const link = document.createElement('a')
      link.download = row.FileName // a标签添加属性
      link.style.display = 'none'
      link.href = URL.createObjectURL(blob)
      document.body.appendChild(link)
      link.click() // 执行下载
      URL.revokeObjectURL(link.href)  // 释放 bolb 对象
      document.body.removeChild(link) // 下载完成移除元素
    }).catch(function (error) {
      console.log(error)
    })
  Row = row
}
const MyPrivacy = (row:{FileId:string, [propName:string]:any}) => {
  dialog_privacy.value = true
  Row = row
}
const FilePrivacy = (row:{FileId:string, [propName:string]:any}) => {
  let formData = new FormData();
  formData.append("FileId",row.FileId);
  formData.append("FilePrivacy",fileprivacy.value);
  axios.post('/file_privacy', formData).then(res => {
    ElMessageBox.alert(res.data.message, '提示', {
      confirmButtonText: 'OK',
      callback: (action: Action) => {
        ElMessage({
          type: 'info',
          message: res.data.message,
        })
      },
    })
    dialog_privacy.value = false
  })
}

const Delete = (row:{FileId:string, [propName:string]:any}) => {
  // console.log(row)
  let formData = new FormData();
  formData.append("FileId",row.FileId);
  formData.append("whether_delete_data",new Boolean(whether_delete_data.value).toString());
  axios.post('/delete', formData).then(res => {
    if(res.data.code != 707){
      ElMessageBox.alert(res.data.message, '提示', {
        confirmButtonText: 'OK',
        callback: (action: Action) => {
          ElMessage({
            type: 'info',
            message: res.data.message,
          })
        },
      })
      let formData = new FormData();
      formData.append("UserName", store.state.log.UserName);
      axios.post('/record', formData).then(res => {
        store.commit('record/UPDATE_tableData', res.data)
        console.log("请求记录成功")
        store.commit('record/UPDATE_GetRecord', true)
      })
    }
    else{
      store.commit('log/UPDATE_LOG', false)
      store.commit('log/UPDATE_EXIT', false)
    }
    if (res.data.message) {
      ElMessage(res.data.message)
    }
  })
  dialog_delete.value = false
}

</script>

<style>
/* .el-table .warning-row {
  --el-table-tr-bg-color: var(--el-color-warning-light-1);
}
.el-table .success-row {
  --el-table-tr-bg-color: var(--el-color-success-light-9);
} */
</style>