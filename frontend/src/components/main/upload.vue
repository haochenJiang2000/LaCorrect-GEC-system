<template>
  <el-upload
    ref="upload"
    v-model:file-list="fileList"
    class="upload-demo"
    :action=my_upload_url
    :limit="1"
    :on-exceed="handleExceed"
    :data="userdata"
    :on-preview="handlePreview"
    :on-remove="handleRemove"
    :before-remove="beforeRemove"
    :before-upload="beforeUpload"
    :on-success="OnSuccess"
  >
    <span>
      <el-select v-model="GEC_type" placeholder="航空航天领域">
        <el-option label="通用领域" value="通用领域" />
        <el-option label="航空航天领域" value="航空航天领域" />
      </el-select>
    </span>  
    <span>
      <el-select v-model="GEC_mode" placeholder="速度优先">
        <el-option label="速度优先" value="速度优先" />
        <el-option label="中度纠错（推荐）" value="中度纠错" />
        <el-option label="深度纠错" value="深度纠错" />
      </el-select>
    </span>  
    <el-button type="primary">点此上传</el-button>
    <template #tip>
      <div class="el-upload__tip">
        上传文件需为doc,docx,pdf格式。
        <div>
          &nbsp
        </div>
        <el-alert style="width: 1000px;" title=
        "每位用户限制提交次数为 3次 / 10min，请勿重复提交; " 
        type="info" :closable="false"/>
        <el-alert style="width: 1000px;" title=
        "一篇常规文档的预测参考时长 速度优先：40-60s | 中度纠错：2-3min | 深度纠错：3-5min" 
        type="info" :closable="false"/>
      </div>
    </template>
  </el-upload>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { ElMessage, ElMessageBox, genFileId } from 'element-plus'
import type { UploadInstance, UploadProps, UploadUserFile, UploadRawFile } from 'element-plus'
import { useStore } from 'vuex'
import type { Action } from 'element-plus'

const store = useStore()
const upload = ref<UploadInstance>()
const GEC_mode = ref("中度纠错")
const GEC_type = ref("航空航天领域")

let xmlhttp = new XMLHttpRequest();
xmlhttp.open('get', './config.json', false);
xmlhttp.send(null);
let my_upload_url = ref(JSON.parse(xmlhttp.responseText).upload_url)

//计算当前时间
var myDate = new Date()
let month = (myDate.getMonth() + 1).toString().padStart(2, '0')
let day = myDate.getDate().toString().padStart(2, '0')
let hour = myDate.getHours().toString().padStart(2, '0')
let minutes = myDate.getMinutes().toString().padStart(2, '0')
let seconed = myDate.getSeconds().toString().padStart(2, '0')

let userdata = {
  Date: myDate.getFullYear() + '-' + month + '-' + day + ' ' + hour + ':' + minutes + ':' + seconed,
  UserName: store.state.log.UserName,
  GEC_mode: GEC_mode.value,
  GEC_type: GEC_type.value,
}

const submitUpload = () => {  
  console.log("上传url:",my_upload_url)
  upload.value!.submit()
}

const handleExceed: UploadProps['onExceed'] = (files) => {
  upload.value!.clearFiles()
  const file = files[0] as UploadRawFile
  file.uid = genFileId()
  upload.value!.handleStart(file)
  submitUpload()
}

const OnSuccess = (res: { [x: string]: any; message: string; FileName: any; Date: any; Status: any }) => {
  if(res.message=="文件上传成功"){
    console.log(res.FileName)
    store.commit('submit/UPDATE_FileName', res.FileName)
    store.commit('submit/UPDATE_FileId', res.FileId)
    const record = {
      FileName:res.FileName,
      Date:res.Date,
      Status:res.Status,
      FileId:res.FileId,
    }
    store.commit('record/UPDATE_Record', record)
  }
  ElMessageBox.alert(res.message, '提示', {
    confirmButtonText: 'OK',
    callback: (action: Action) => {
      ElMessage({
        type: 'info',
        message: res.message,
      })
    },
  })
}

const fileList = ref<UploadUserFile[]>([])

const handleRemove: UploadProps['onRemove'] = (file, uploadFiles) => {
  console.log(file, uploadFiles)
}

const beforeUpload: UploadProps['beforeUpload'] = (uploadFile) => {
  userdata = {
    Date: myDate.getFullYear() + '-' + month + '-' + day + ' ' + hour + ':' + minutes + ':' + seconed,
    UserName: store.state.log.UserName,
    GEC_mode: GEC_mode.value,
    GEC_type: GEC_type.value,
  }
}

const handlePreview: UploadProps['onPreview'] = (uploadFile) => {
  console.log(uploadFile)
}

const beforeRemove: UploadProps['beforeRemove'] = (uploadFile) => {
  return ElMessageBox.confirm(
    `是否删除文件 ${uploadFile.name} ?`
  ).then(
    () => true,
    () => false
  )
}

</script>

<style scoped>
</style>