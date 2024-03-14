<template>
  <div class="sidebar_top">
    <div v-if="LoggedIn">
      <div>
        <el-image
          style="width: 100px; height: 60px"
          :src="url"
        />
          欢迎！{{Username}}
      </div>
    </div>
    <div v-else>
      <el-image
        style="width: 100px; height: 60px"
        :src="url"
      />
    </div>
    <div v-if="!LoggedIn">
      <el-button
        type='primary'
        @click="dialogTableVisible = true"
      >
        登录/注册
      </el-button>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp

      <el-dialog
        v-model="dialogTableVisible"
        width="800px"
        destroy-on-close
        center
      >
        <Login
        v-model:dialogTableVisible="dialogTableVisible"
        ></Login>
      </el-dialog>
    </div>
    <div v-else>
      <el-button
        class="button" text
        @click="toInfo"
      >
        用户信息
      </el-button>
      <el-button
        class="button" text
        @click="exit"
      >
        退出
      </el-button>
      &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
    </div>
  </div>
</template>

<script lang='ts' setup>
  // import Login from '../login/log_reg.vue'
  import router from '@/router';
  import Login from '../login/register.vue'
  import { ref } from 'vue'
  import { computed } from 'vue'
  import { useStore } from 'vuex'
  import { ElMessage, ElMessageBox } from 'element-plus'
  import type { Action } from 'element-plus'

  const url = require('../../../src/assets/LA.jpg')

  let store = useStore()
  const dialogTableVisible = ref(true)
  let LoggedIn = computed(() => store.state.log.LoggedIn || false)
  let Username = computed(() => store.state.log.UserName)
  if(LoggedIn.value){
    dialogTableVisible.value = false
  }
  else{
    dialogTableVisible.value = true
  }

  const exit = () => {
    console.log("注销成功！")
    ElMessageBox.alert('注销成功！', '提示', {
      confirmButtonText: 'OK',
      callback: (action: Action) => {
        ElMessage({
          type: 'info',
          message: `注销成功！`,
        })
      },
    })
    store.commit('log/UPDATE_LOG', false)
    store.commit('log/UPDATE_EXIT', false)
    if ((window.getSelection() as Selection).toString() === ''){
        router.push({ path: "/submit"})
      }
  }

  const toInfo = () => {
    if ((window.getSelection() as Selection).toString() === ''){
        router.push({ path: "/info"})
      }
  }

</script>

