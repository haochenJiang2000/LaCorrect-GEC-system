<template>
  <div class="common-layout">
    <el-container>
      <el-header>
        <Head></Head>
      </el-header>
      <el-container>
        <el-aside width="200px">
          <Menu></Menu>
        </el-aside>
        <el-main>
          <router-view></router-view>
        </el-main>
      </el-container>
    </el-container>
    <!-- <div
      style="position:fixed;left: 650px;bottom:0;margin:10 auto;width:100%;color: #5c6b77"
    >
      <a
        target="_blank"
        style="color: #5c6b77"
        href="https://beian.miit.gov.cn/"
        >苏ICP备2021041749号-2</a
      >
    </div> -->
  </div>

</template>

<script lang="ts">
import { Options, Vue } from 'vue-class-component';
import Menu from './components/head/menu.vue'
import Head from './components/head/head.vue'
import Login from './components/login/register.vue'
import MyPrivacy from './components/login/privacy.vue'
import { createDOMCompilerError } from '@vue/compiler-dom';
import store from './store';

@Options({
  components: {
    Menu,
    Head,
    Login,
    MyPrivacy,
  },
})
export default class App extends Vue {}
</script>

<style>
html,
body {
  height: 100%;
  font-family: "思源黑体 Regular";
}
</style>
